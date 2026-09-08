//! MTP speculative decoding for the in-process llama.cpp engine.
//!
//! Measured on a Jetson Orin Nano 8GB with gemma-4-E4B-it-qat: llama.cpp's MTP
//! gives 47.7 tok/s against 15.8 without, at 87% draft acceptance. This brings
//! that in-process instead of running a `llama-server` sidecar.
//!
//! # The invariant this module exists to protect
//!
//! Speculative decoding is only sound if the emitted tokens are EXACTLY the
//! tokens the target model would have emitted alone. The drafter proposes; the
//! target verifies; anything the target rejects must leave no trace. Two things
//! make that easy to get subtly wrong, and both fail silently:
//!
//! 1. **Rejected drafts stay in the KV cache.** The draft is decoded on the
//!    target to obtain its logits, which writes those positions. Whatever is not
//!    accepted must be removed again, or [`SessionKv`](super::inference_engine)'s
//!    core invariant -- that its token ledger is a PREFIX of KV contents -- is
//!    broken, and every later `ReusePrefix` resumes from a cache that disagrees
//!    with the ledger.
//! 2. **Off-by-one in the accept count.** `common_speculative_accept` counts
//!    accepted DRAFT tokens, not emitted tokens. The token sampled from the last
//!    verified position is emitted but is not a draft acceptance.
//!
//! Neither produces an error. Both produce fluent, wrong output. So
//! `greedy_equivalence` in the tests asserts the only property that actually
//! matters: at temperature 0, output with speculation must be byte-identical to
//! output without it.

use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::speculative::{MtpSpeculative, MtpSpeculativeParams};
use llama_cpp_2::token::LlamaToken;

use goose_provider_types::errors::ProviderError;

/// A target context with an MTP drafter attached.
pub struct MtpSession {
    spec: MtpSpeculative<'static>,
    /// Draft depth actually requested, so telemetry can report acceptance
    /// against what was asked for rather than against a constant.
    n_max: usize,
    /// Cumulative, for `DraftStats`.
    pub drafted: u32,
    pub accepted: u32,
}

impl MtpSession {
    /// Attach a drafter to a target context.
    ///
    /// Both contexts are consumed: that is `MtpSpeculative`'s contract, and it
    /// is why `SessionCtx` exists.
    pub fn new(
        target: LlamaContext<'static>,
        draft: LlamaContext<'static>,
        n_max: i32,
        p_min: f32,
    ) -> Result<Self, ProviderError> {
        let spec = MtpSpeculative::new(
            target,
            draft,
            MtpSpeculativeParams {
                n_max,
                n_min: 0,
                p_min,
            },
        )
        .map_err(|e| {
            ProviderError::ExecutionError(format!("failed to init MTP speculative decoding: {e}"))
        })?;
        Ok(Self {
            spec,
            n_max: n_max.max(0) as usize,
            drafted: 0,
            accepted: 0,
        })
    }

    pub fn target(&self) -> &LlamaContext<'static> {
        self.spec.target_context()
    }

    pub fn target_mut(&mut self) -> &mut LlamaContext<'static> {
        self.spec.target_context_mut()
    }

    /// Tell the drafter which prompt the target has been prefilled with.
    ///
    /// Must be called after every prefill, including a `ReusePrefix` resume: the
    /// drafter tracks its own position, and a resume that skips this leaves it
    /// drafting against a prompt the target is no longer at.
    pub fn begin(&mut self, prompt: &[LlamaToken]) -> Result<(), ProviderError> {
        self.spec
            .begin(prompt)
            .map_err(|e| ProviderError::ExecutionError(format!("MTP begin failed: {e}")))
    }

    /// Propose up to `n_max` continuations of `id_last`.
    ///
    /// An empty result is normal and not an error -- the drafter declines when
    /// its own confidence is below `p_min`.
    pub fn draft(
        &mut self,
        n_past: i32,
        id_last: LlamaToken,
        prompt: &[LlamaToken],
    ) -> Result<Vec<LlamaToken>, ProviderError> {
        if self.n_max == 0 {
            return Ok(Vec::new());
        }
        let out = self
            .spec
            .draft(n_past, id_last, prompt)
            .map_err(|e| ProviderError::ExecutionError(format!("MTP draft failed: {e}")))?;
        self.drafted += out.len() as u32;
        Ok(out)
    }

    /// Report how many DRAFT tokens the target endorsed.
    ///
    /// Not how many tokens were emitted: the token sampled from the final
    /// verified position is emitted too, and counting it here would tell the
    /// drafter its proposals were better than they were.
    pub fn accept(&mut self, n_accepted: usize) -> Result<(), ProviderError> {
        self.accepted += n_accepted as u32;
        self.spec
            .accept(n_accepted as u16)
            .map_err(|e| ProviderError::ExecutionError(format!("MTP accept failed: {e}")))
    }

    pub fn acceptance_rate(&self) -> Option<f32> {
        (self.drafted > 0).then(|| self.accepted as f32 / self.drafted as f32)
    }
}

/// One speculative step.
///
/// Returns the tokens the TARGET endorsed, in order, and whether generation
/// should stop because an end-of-generation token was reached. The caller emits
/// exactly these and nothing else.
///
/// `n_past` is the position `id_last` occupies. On entry the target's KV holds
/// everything up to and including `id_last`.
pub struct SpecStep {
    pub tokens: Vec<LlamaToken>,
    pub hit_eog: bool,
}

pub fn speculate(
    mtp: &mut MtpSession,
    model: &LlamaModel,
    sampler: &mut llama_cpp_2::sampling::LlamaSampler,
    n_past: i32,
    id_last: LlamaToken,
    prompt: &[LlamaToken],
) -> Result<SpecStep, ProviderError> {
    let draft = mtp.draft(n_past, id_last, prompt)?;
    if draft.is_empty() {
        // Nothing proposed: fall back to a plain single-token decode so the
        // caller's loop shape does not change.
        let one = [id_last];
        let mut batch = LlamaBatch::get_one(&one)
            .map_err(|e| ProviderError::ExecutionError(format!("batch: {e}")))?;
        mtp.target_mut()
            .decode(&mut batch)
            .map_err(|e| ProviderError::ExecutionError(format!("decode: {e}")))?;
        mtp.accept(0)?;
        return Ok(SpecStep {
            tokens: Vec::new(),
            hit_eog: false,
        });
    }

    // Decode `id_last` followed by the whole draft in ONE batch, asking for
    // logits at every position. Position i's logits are the target's opinion of
    // what should follow draft[i-1], which is what verification compares against.
    let mut batch = LlamaBatch::new(draft.len() + 1, 1);
    let mut seq = Vec::with_capacity(draft.len() + 1);
    seq.push(id_last);
    seq.extend_from_slice(&draft);
    for (i, tok) in seq.iter().enumerate() {
        batch
            .add(*tok, n_past + i as i32, &[0], true)
            .map_err(|e| ProviderError::ExecutionError(format!("batch add: {e}")))?;
    }
    mtp.target_mut()
        .decode(&mut batch)
        .map_err(|e| ProviderError::ExecutionError(format!("speculative decode: {e}")))?;

    // Verify. The target's choice at position i either matches draft[i] -- in
    // which case the draft saved a forward pass -- or it does not, and the
    // target's own token wins and drafting stops for this step.
    let mut emitted = Vec::with_capacity(draft.len() + 1);
    let mut n_accepted = 0usize;
    let mut hit_eog = false;
    for i in 0..seq.len() {
        let tok = sampler.sample(mtp.target_mut(), i as i32);
        sampler.accept(tok);
        if model.is_eog_token(tok) {
            hit_eog = true;
            break;
        }
        emitted.push(tok);
        if i < draft.len() && tok == draft[i] {
            n_accepted += 1;
        } else {
            // Divergence (or the final verified position): stop here. Everything
            // after this in the batch was speculation the target did not endorse.
            break;
        }
    }

    // Roll the unendorsed tail out of the KV cache.
    //
    // The batch wrote `seq.len()` positions. Only `emitted.len()` of them are
    // real. Without this the token ledger stops being a prefix of KV contents
    // and every later ReusePrefix resumes against a cache that disagrees with it.
    let keep_upto = n_past + emitted.len() as i32;
    let wrote_upto = n_past + seq.len() as i32;
    if keep_upto < wrote_upto {
        mtp.target_mut()
            .clear_kv_cache_seq(Some(0), Some(keep_upto as u32), None)
            .map_err(|e| {
                ProviderError::ExecutionError(format!("failed to roll back rejected draft: {e}"))
            })?;
    }

    mtp.accept(n_accepted)?;
    Ok(SpecStep {
        tokens: emitted,
        hit_eog,
    })
}

#[cfg(test)]
mod tests {
    //! The only test that settles whether speculation is correct needs a real
    //! model and a real drafter, so it is `#[ignore]`d and run explicitly:
    //!
    //! ```text
    //! GIAP_TEST_GGUF=<target.gguf> GIAP_TEST_DRAFT=<drafter.gguf> \
    //!   cargo test -p goose-local-inference --lib mtp -- --ignored --nocapture
    //! ```
    //!
    //! Everything else about MTP can look right while being wrong: a drafter
    //! pointed at a stale prompt still returns tokens, a rejected draft left in
    //! the KV cache still decodes, an off-by-one accept still produces fluent
    //! text. What none of those survive is this: at temperature 0, speculative
    //! decoding is DEFINED to emit exactly what the target would have emitted
    //! alone. Byte-identical, or the implementation is wrong.

    /// Greedy equivalence: same prompt, same seed, speculation on and off.
    ///
    /// Deliberately compares the token IDs rather than the decoded string --
    /// two different token sequences can decode to the same text, and that
    /// would hide precisely the off-by-one this is guarding.
    #[test]
    #[ignore = "needs GIAP_TEST_GGUF and GIAP_TEST_DRAFT pointing at real models"]
    fn greedy_output_is_identical_with_and_without_speculation() {
        let (Ok(target), Ok(draft)) = (
            std::env::var("GIAP_TEST_GGUF"),
            std::env::var("GIAP_TEST_DRAFT"),
        ) else {
            eprintln!("set GIAP_TEST_GGUF and GIAP_TEST_DRAFT to run this");
            return;
        };
        eprintln!("target={target} draft={draft}");
        eprintln!(
            "NOT YET IMPLEMENTED: needs a backend + model load harness. Until this \
             body exists and passes on the device, in-process MTP is UNVERIFIED and \
             must not be enabled by default."
        );
        panic!("greedy-equivalence harness not written");
    }
}
