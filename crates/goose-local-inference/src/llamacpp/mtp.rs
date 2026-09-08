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
    /// Cumulative microseconds per phase of [`speculate`].
    ///
    /// Acceptance alone cannot explain a speculative run that comes out SLOWER:
    /// drafting can be working perfectly and still cost more than the target
    /// decodes it saves. These four say which phase paid.
    pub draft_us: u64,
    pub verify_us: u64,
    pub process_us: u64,
    pub rollback_us: u64,
    pub steps: u32,
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
            draft_us: 0,
            verify_us: 0,
            process_us: 0,
            rollback_us: 0,
            steps: 0,
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
    /// Feed the drafter a batch the target has just decoded.
    ///
    /// llama.cpp's server calls `common_speculative_process` after EVERY decode
    /// on the target, with that same batch. Skipping it leaves the drafter's
    /// state diverged from the target's from the first token, which surfaces as
    /// `llama_decode[0] returned -1` inside the drafter rather than as anything
    /// pointing at the real cause.
    pub fn process(&mut self, batch: &LlamaBatch<'_>) -> Result<(), ProviderError> {
        self.spec
            .process(batch)
            .map_err(|e| ProviderError::ExecutionError(format!("MTP process failed: {e}")))
    }

    pub fn accept(&mut self, n_accepted: usize) -> Result<(), ProviderError> {
        self.accepted += n_accepted as u32;
        self.spec
            .accept(n_accepted as u16)
            .map_err(|e| ProviderError::ExecutionError(format!("MTP accept failed: {e}")))
    }
}

/// One speculative step.
///
/// # Invariant
///
/// On entry `id_last` is NOT yet in the KV cache -- this call decodes it, at
/// position `n_past`. On return the cache holds exactly `id_last` plus the
/// accepted drafts, and `next` is again undecoded, ready to be the following
/// step's `id_last`.
///
/// Holding that invariant is the whole correctness argument. The first version
/// of this let the caller re-sample after the step, which sampled from logits
/// belonging to a token that had never been decoded, and it kept one KV position
/// too few, deleting an accepted token. Neither errored; both produced fluent
/// output that diverged from the target's within three tokens.
pub struct SpecStep {
    /// Tokens the target endorsed, in order. Already decoded.
    pub tokens: Vec<LlamaToken>,
    /// The next token, sampled but NOT decoded. `None` at end of generation.
    pub next: Option<LlamaToken>,
}

pub fn speculate(
    mtp: &mut MtpSession,
    model: &LlamaModel,
    sampler: &mut llama_cpp_2::sampling::LlamaSampler,
    n_past: i32,
    id_last: LlamaToken,
    prompt: &[LlamaToken],
) -> Result<SpecStep, ProviderError> {
    let t = std::time::Instant::now();
    let draft = mtp.draft(n_past, id_last, prompt)?;
    mtp.draft_us += t.elapsed().as_micros() as u64;

    // `id_last` first, then the whole draft, logits requested at every position.
    // Position i's logits are the target's opinion of what follows seq[i].
    let mut seq = Vec::with_capacity(draft.len() + 1);
    seq.push(id_last);
    seq.extend_from_slice(&draft);
    let mut batch = LlamaBatch::new(seq.len(), 1);
    for (i, tok) in seq.iter().enumerate() {
        batch
            .add(*tok, n_past + i as i32, &[0], true)
            .map_err(|e| ProviderError::ExecutionError(format!("batch add: {e}")))?;
    }
    let t = std::time::Instant::now();
    mtp.target_mut()
        .decode(&mut batch)
        .map_err(|e| ProviderError::ExecutionError(format!("speculative decode: {e}")))?;
    // Metal and CUDA return from `decode` before the graph has run, so without
    // this the target's forward pass is billed to whichever phase reads an
    // output first -- which made `verify` read 1.2 ms against a 37.7 ms plain
    // decode, and `process` absorb the difference.
    mtp.target_mut().synchronize();
    mtp.verify_us += t.elapsed().as_micros() as u64;

    let t = std::time::Instant::now();
    mtp.process(&batch)?;
    mtp.process_us += t.elapsed().as_micros() as u64;

    // Verify: walk the draft while the target agrees with it.
    let mut accepted = Vec::new();
    let mut n_acc = 0usize;
    let mut next = sampler.sample(mtp.target_mut(), 0);
    sampler.accept(next);
    while n_acc < draft.len() && next == draft[n_acc] && !model.is_eog_token(next) {
        accepted.push(next);
        n_acc += 1;
        next = sampler.sample(mtp.target_mut(), n_acc as i32);
        sampler.accept(next);
    }

    // Drop the unendorsed tail. The cache legitimately holds `id_last` at
    // `n_past` plus `n_acc` accepted drafts after it, so the first position to
    // remove is n_past + n_acc + 1. Using n_past + n_acc here deleted an
    // accepted token and desynchronised the ledger from the cache.
    let t = std::time::Instant::now();
    let first_stale = n_past + n_acc as i32 + 1;
    if (first_stale as usize) < (n_past as usize) + seq.len() {
        mtp.target_mut()
            .clear_kv_cache_seq(Some(0), Some(first_stale as u32), None)
            .map_err(|e| {
                ProviderError::ExecutionError(format!("failed to roll back rejected draft: {e}"))
            })?;
    }

    mtp.rollback_us += t.elapsed().as_micros() as u64;
    mtp.steps += 1;

    mtp.accept(n_acc)?;
    Ok(SpecStep {
        tokens: accepted,
        next: (!model.is_eog_token(next)).then_some(next),
    })
}

#[cfg(test)]
mod tests {
    //! Greedy equivalence is the only test that settles whether speculation is
    //! correct, and it needs real weights, so it is `#[ignore]`d:
    //!
    //! ```text
    //! cargo test -p goose-local-inference --lib mtp -- --ignored --nocapture
    //! ```
    //!
    //! Everything about MTP can look right while being wrong. A drafter pointed
    //! at a stale prompt still returns tokens. A rejected draft left in the KV
    //! cache still decodes. An off-by-one accept still produces fluent text.
    //! None of those raise an error. What none of them survive is this: at
    //! temperature 0, speculative decoding is DEFINED to emit exactly what the
    //! target would have emitted alone.
    use super::*;
    use crate::llamacpp::inference_engine::tests::shared_backend;
    use crate::llamacpp::inference_engine::{generation_loop, SessionKv, TokenAction};
    use crate::local_model_registry::{ModelSettings, SamplingConfig};
    use llama_cpp_2::model::params::LlamaModelParams;
    use llama_cpp_2::model::AddBos;
    use std::sync::Arc;

    /// What one speculative run cost, by phase.
    struct DraftReport {
        drafted: u32,
        accepted: u32,
        steps: u32,
        draft_us: u64,
        verify_us: u64,
        process_us: u64,
        rollback_us: u64,
    }

    /// Serialises the live tests against each other.
    ///
    /// Both load 4 GB of weights and saturate the GPU, and cargo runs tests in
    /// parallel by default. Sharing the device does not fail either of them --
    /// it halves both their numbers, which is worse: the first run of the pair
    /// reported plain decode at 12.3 tok/s against 25.7 measured alone, and a
    /// 0.52x "regression" that was pure contention.
    fn exclusive_gpu() -> std::sync::MutexGuard<'static, ()> {
        static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());
        GPU.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Target and drafter, hard-linked into a temp dir.
    ///
    /// Hard-linked and never symlinked: a symlinked models directory has twice
    /// caused a scratch run to reach through and destroy a real pond's weights.
    fn live_pair() -> Option<(std::path::PathBuf, std::path::PathBuf)> {
        let home = std::path::PathBuf::from(std::env::var_os("HOME")?);
        let roots = [
            home.join("Library/Application Support/goose-in-a-pond/models/gguf"),
            home.join(".local/share/goose-in-a-pond/models/gguf"),
        ];
        let target = std::env::var_os("GIAP_TEST_GGUF").map(std::path::PathBuf::from);
        let draft = std::env::var_os("GIAP_TEST_DRAFT").map(std::path::PathBuf::from);
        let target = target.or_else(|| {
            roots
                .iter()
                .map(|r| r.join("gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf"))
                .find(|p| p.is_file())
        })?;
        let draft = draft.or_else(|| {
            roots
                .iter()
                .map(|r| r.join("mtp-gemma-4-E4B-it.gguf"))
                .find(|p| p.is_file())
        })?;
        (target.is_file() && draft.is_file()).then_some((target, draft))
    }

    /// Greedy settings. Temperature 0 with a fixed seed, because the equivalence
    /// claim only holds for deterministic sampling.
    /// Greedy, with the context configuration the Jetson actually ships.
    ///
    /// `ModelSettings::default()` leaves every one of these `None`, which means
    /// no flash attention and an f16 KV cache -- and a quantised V cache is
    /// refused outright without flash attention. The 2.8x this work is chasing
    /// was measured on llama-server run with `-fa on -ctk q8_0 -ctv q8_0 -b 512
    /// -ub 128`, so measuring against the defaults compares two different
    /// configurations and blames the difference on the drafter. Flash attention
    /// in particular changes what a batched verify costs, which is the one thing
    /// speculation is buying.
    fn greedy(n_max: Option<i32>) -> ModelSettings {
        ModelSettings {
            sampling: SamplingConfig::Greedy,
            draft_n_max: n_max,
            draft_p_min: Some(0.0),
            flash_attention: Some(true),
            type_k: Some("q8_0".to_string()),
            type_v: Some("q8_0".to_string()),
            n_batch: Some(512),
            n_ubatch: Some(128),
            ..Default::default()
        }
    }

    fn generate(
        model: &Arc<llama_cpp_2::model::LlamaModel>,
        draft: Option<&Arc<llama_cpp_2::model::LlamaModel>>,
        settings: &ModelSettings,
        prompt: &[LlamaToken],
        n_ctx: u32,
        budget: usize,
    ) -> (Vec<LlamaToken>, Option<DraftReport>) {
        let backend = shared_backend();
        let mut kv = SessionKv::create(model, draft, backend, n_ctx, settings).expect("context");
        crate::llamacpp::inference_engine::decode_tokens_for_test(&mut kv, prompt)
            .expect("prefill");
        let mut decoded = Vec::new();
        let mut capped = settings.clone();
        capped.max_output_tokens = Some(budget);
        let _ = generation_loop(
            model,
            kv.session_ctx_mut(),
            &capped,
            prompt,
            prompt.len(),
            n_ctx as usize,
            &mut decoded,
            |_| Ok(TokenAction::Continue),
        );
        let stats = kv.session_ctx_mut().mtp_mut().map(|m| DraftReport {
            drafted: m.drafted,
            accepted: m.accepted,
            steps: m.steps,
            draft_us: m.draft_us,
            verify_us: m.verify_us,
            process_us: m.process_us,
            rollback_us: m.rollback_us,
        });
        (decoded, stats)
    }

    /// What one target forward pass costs as a function of batch size.
    ///
    /// This curve, and nothing about the drafter, decides whether speculative
    /// decoding can pay on a given machine. A step that verifies `d` drafts
    /// costs `T(d+1)` plus the drafter and emits `1+a` tokens, against
    /// `(1+a) * T(1)` for plain decoding. So the win requires `T(n)` to be
    /// nearly flat in `n` -- true when the forward pass is bandwidth-bound,
    /// because the weights are streamed once however many positions ride along.
    ///
    /// Run it on any machine where MTP disappoints before touching the drafter:
    /// a steep curve here means there was never a speedup available, whatever
    /// the acceptance rate says.
    #[test]
    #[ignore = "requires a real GGUF; run with --ignored"]
    fn target_forward_cost_by_batch_size() {
        let _gpu = exclusive_gpu();
        let Some((target_src, _)) = live_pair() else {
            eprintln!("skipping: no target found");
            return;
        };
        let tmp = tempfile::tempdir().expect("tempdir");
        let target_path = tmp.path().join("target.gguf");
        std::fs::hard_link(&target_src, &target_path).expect("hard link target");

        let backend = shared_backend();
        let params = LlamaModelParams::default().with_n_gpu_layers(99);
        let model = Arc::new(
            llama_cpp_2::model::LlamaModel::load_from_file(
                backend.llama_backend(),
                &target_path,
                &params,
            )
            .expect("load target"),
        );

        let n_ctx: u32 = 2048;
        let settings = greedy(None);
        let mut kv = SessionKv::create(&model, None, backend, n_ctx, &settings).expect("context");
        let filler = model.token_eos();

        let mut pos = 0i32;
        let mut cost = |n: usize, pos: &mut i32, reps: usize| -> f64 {
            let mut total = 0.0;
            for _ in 0..reps {
                let mut batch = LlamaBatch::new(n, 1);
                for i in 0..n {
                    batch.add(filler, *pos + i as i32, &[0], true).expect("add");
                }
                let t = std::time::Instant::now();
                let ctx = kv.session_ctx_mut().ctx_mut();
                ctx.decode(&mut batch).expect("decode");
                ctx.synchronize();
                total += t.elapsed().as_secs_f64() * 1000.0;
                *pos += n as i32;
            }
            total / reps as f64
        };

        // Each distinct batch shape builds its own graph on first use, and
        // billing that to the first timed call made T(9) come out CHEAPER than
        // T(5). Warm every size before timing it, not just the first.
        let mut warmed = |n: usize, pos: &mut i32| {
            cost(n, pos, 2);
            cost(n, pos, 8)
        };
        let t1 = warmed(1, &mut pos);
        eprintln!("target forward, logits at every position:");
        eprintln!("   n     ms   ms/n   T(n)/T(1)   verdict at full acceptance");
        for n in [1usize, 2, 3, 4, 5, 6, 7, 8, 9, 13, 17] {
            let t = warmed(n, &mut pos);
            let per = t / n as f64;
            eprintln!(
                "  {n:>2}  {t:>5.1}  {per:>5.1}      {:>5.2}x   {}",
                t / t1,
                if per < t1 {
                    format!("{:.2}x headroom", t1 / per)
                } else {
                    "no headroom -- batching does not amortise".to_string()
                }
            );
        }
        eprintln!(
            "\n  A draft depth of d verifies a batch of d+1. Speculation can only\n  \
             pay where the ms/n column falls below {t1:.1} ms."
        );
    }

    /// The token IDs must match, not the decoded text.
    ///
    /// Two different token sequences can decode to the same string, and that is
    /// exactly where an off-by-one in the accept count would hide.
    #[test]
    #[ignore = "requires a real target GGUF and MTP drafter; run with --ignored"]
    fn greedy_output_is_identical_with_and_without_speculation() {
        let _gpu = exclusive_gpu();
        let Some((target_src, draft_src)) = live_pair() else {
            eprintln!("skipping: no target+drafter pair found");
            return;
        };
        let tmp = tempfile::tempdir().expect("tempdir");
        let target_path = tmp.path().join("target.gguf");
        let draft_path = tmp.path().join("draft.gguf");
        std::fs::hard_link(&target_src, &target_path).expect("hard link target");
        std::fs::hard_link(&draft_src, &draft_path).expect("hard link drafter");
        eprintln!(
            "target: {}\ndraft:  {}",
            target_src.display(),
            draft_src.display()
        );

        let backend = shared_backend();
        // Offloaded, because production is: `apply_jetson_settings` stamps
        // n_gpu_layers 99 and `LlamaModelParams::default()` is 0. A CPU forward
        // pass costs in proportion to the batch, so speculation cannot win on
        // one -- the first version of this test defaulted, and measured 0.80x
        // for that reason alone.
        let params = LlamaModelParams::default().with_n_gpu_layers(99);
        let load = |path: &std::path::Path| {
            Arc::new(
                llama_cpp_2::model::LlamaModel::load_from_file(
                    backend.llama_backend(),
                    path,
                    &params,
                )
                .expect("load model"),
            )
        };
        let model = load(&target_path);
        let drafter = load(&draft_path);

        let n_ctx: u32 = 2048;
        let budget = 64;
        let prompt = model
            .str_to_token(
                "List the first ten prime numbers, separated by commas.",
                AddBos::Always,
            )
            .expect("tokenize");

        // Timed, and swept over draft depth, because whether the llama-server
        // speedup survives moving into GIAP's loop is the question the bake-off
        // could not answer -- and because correctness at one depth does not
        // imply it at another: the accept/rollback arithmetic is indexed by the
        // draft length.
        let t0 = std::time::Instant::now();
        let (plain, _) = generate(&model, None, &greedy(None), &prompt, n_ctx, budget);
        let plain_ms = t0.elapsed().as_millis().max(1) as f64;
        let per_token = plain_ms / plain.len().max(1) as f64;
        eprintln!(
            "plain: {} tokens in {:.0} ms ({:.1} tok/s, {:.1} ms/token)",
            plain.len(),
            plain_ms,
            plain.len() as f64 * 1000.0 / plain_ms,
            per_token
        );
        assert!(!plain.is_empty(), "the control run produced nothing");

        for depth in [1i32, 4, 8, 12] {
            let t1 = std::time::Instant::now();
            let (spec, stats) = generate(
                &model,
                Some(&drafter),
                &greedy(Some(depth)),
                &prompt,
                n_ctx,
                budget,
            );
            let spec_ms = t1.elapsed().as_millis().max(1) as f64;
            let r = stats.expect("a drafter was supplied, so there is an MTP session");
            let steps = r.steps.max(1) as f64;
            eprintln!(
                "depth {depth:>2}: {:>5.0} ms  {:>4.2}x  |  {}/{} drafts kept ({:>3.0}%), \
                 {:.2} tok/step  |  draft {:>5.1} + verify {:>5.1} ms/step",
                spec_ms,
                plain_ms / spec_ms,
                r.accepted,
                r.drafted,
                r.accepted as f64 * 100.0 / r.drafted.max(1) as f64,
                spec.len() as f64 / steps,
                r.draft_us as f64 / 1000.0 / steps,
                r.verify_us as f64 / 1000.0 / steps,
            );
            // Both should be ~0. A non-zero `process` means the drafter is NOT
            // sharing the target's KV cache and is paying for a catch-up decode
            // per step, which is the difference between MTP being worth having
            // and not; a non-zero `rollback` means drafts are being rejected
            // often enough for the seq_rm to matter.
            eprintln!(
                "          process {:.1} + rollback {:.1} ms/step",
                r.process_us as f64 / 1000.0 / steps,
                r.rollback_us as f64 / 1000.0 / steps,
            );

            assert_eq!(
                plain,
                spec,
                "speculative output diverged from the target at draft depth {depth} \
                 ({} tokens vs {}). At temperature 0 these are DEFINED to be equal; a \
                 difference means either a draft was endorsed that the target would not \
                 have emitted, or the KV rollback dropped one it would have.",
                plain.len(),
                spec.len(),
            );
        }
    }
}
