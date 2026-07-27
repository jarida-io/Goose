use crate::backend::LocalInferenceBackend;
use crate::local_model_registry::ModelSettings;
use crate::multimodal::ExtractedImage;
use goose_provider_types::errors::ProviderError;
use goose_provider_types::request_log::{LoggerHandleExt, RequestLogHandle};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::{AddBos, ChatTemplateResult, LlamaChatTemplate, LlamaModel};
use llama_cpp_2::mtmd::{MtmdBitmap, MtmdContext, MtmdInputText};
use llama_cpp_2::openai::OpenAIChatTemplateParams;
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use std::num::NonZeroU32;
use std::sync::Arc;

use super::super::StreamSender;
use super::LlamaCppBackend;

/// Shortest shared prompt prefix worth a partial KV cache removal. Below this
/// the bookkeeping costs more than the decode it saves.
const REUSE_MIN_TOKENS: usize = 256;

pub(super) struct GenerationContext<'a> {
    pub model: &'a Arc<LlamaModel>,
    pub mtmd_ctx: Option<&'a MtmdContext>,
    pub session: &'a mut Option<SessionKv>,
    pub backend: &'a LlamaCppBackend,
    pub template: &'a LlamaChatTemplate,
    pub settings: &'a ModelSettings,
    pub context_limit: usize,
    pub model_name: String,
    pub message_id: &'a str,
    pub tx: &'a StreamSender,
    pub log: &'a mut Option<Box<dyn RequestLogHandle>>,
    pub images: &'a [ExtractedImage],
    /// Cold-load time for this request's model, when the load happened in this
    /// request. None when the model was already resident.
    pub model_load_ms: Option<u64>,
}

pub(super) struct LoadedModel {
    pub model: Arc<LlamaModel>,
    pub templates: LoadedChatTemplates,
    /// Multimodal context for vision models. None for text-only models.
    pub mtmd_ctx: Option<MtmdContext>,
    /// Generation context retained between requests for prompt prefix reuse.
    /// Dropped with the rest of the model when the slot is evicted.
    pub session: Option<SessionKv>,
}

pub(super) struct LoadedChatTemplates {
    pub default: Option<LlamaChatTemplate>,
    pub tool_use: Option<LlamaChatTemplate>,
    pub force_default: bool,
}

/// A llama context retained across generations, paired with the token sequence
/// it holds in the KV cache of sequence 0.
///
/// Invariant: `tokens` is a *prefix* of the KV cache contents — position `i`
/// of sequence 0 holds `tokens[i]` for every `i < tokens.len()`. The cache may
/// hold further positions beyond `tokens.len()`; every reuse path removes
/// everything from its resume position onwards, so unrecorded trailing
/// positions can never be attended to.
pub(super) struct SessionKv {
    /// Declared before `_model`: fields drop in declaration order, so the
    /// context is destroyed before the model allocation it points into.
    ctx: LlamaContext<'static>,
    tokens: Vec<LlamaToken>,
    /// The `n_ctx` this context was *requested* with. Compared against the
    /// effective context of an incoming request; llama.cpp may round the
    /// realised `n_ctx` up, so the requested value is the stable identity.
    n_ctx: u32,
    _model: Arc<LlamaModel>,
}

// SAFETY: `LlamaContext` is `!Send` only because it holds a raw
// `NonNull<llama_context>`. A llama.cpp context has no thread affinity: it
// keeps no thread-local state and reads its thread pool out of the context on
// every decode, so it may be moved between threads provided access is never
// concurrent. Every access to a `SessionKv` happens inside
// `LocalInferenceBackend::generate`, which `InferenceRuntime::generate` calls
// while holding the owning model slot's mutex, so accesses are serialized.
// `LlamaModel` and `MtmdContext` are marked `Send` upstream on the same basis.
unsafe impl Send for SessionKv {}

impl SessionKv {
    fn create(
        model: &Arc<LlamaModel>,
        backend: &LlamaCppBackend,
        n_ctx: u32,
        settings: &ModelSettings,
    ) -> Result<Self, ProviderError> {
        let model = Arc::clone(model);
        let ctx = model
            .new_context(
                backend.llama_backend(),
                build_context_params(n_ctx, settings),
            )
            .map_err(|e| ProviderError::ExecutionError(format!("Failed to create context: {e}")))?;

        // SAFETY: `ctx` borrows the `LlamaModel` owned by the `Arc` allocation,
        // whose address is stable for as long as any strong handle exists.
        // Erasing the lifetime is sound because it is immediately re-tied to
        // `_model`, a strong handle stored in this same struct and dropped
        // *after* `ctx` by field declaration order. The context can therefore
        // never outlive the model it points at.
        let ctx = unsafe { std::mem::transmute::<LlamaContext<'_>, LlamaContext<'static>>(ctx) };

        Ok(Self {
            ctx,
            tokens: Vec::new(),
            n_ctx,
            _model: model,
        })
    }

    pub(super) fn context_mut(&mut self) -> &mut LlamaContext<'static> {
        &mut self.ctx
    }

    pub(super) fn record_generated(&mut self, tokens: &[LlamaToken]) {
        self.tokens.extend_from_slice(tokens);
    }
}

/// Context-window size of a retained KV cache, or 0 when nothing is retained.
fn retained_kv_tokens(session: &Option<SessionKv>) -> usize {
    session.as_ref().map_or(0, |kv| kv.n_ctx as usize)
}

pub(super) struct PreparedGeneration {
    pub template_result: ChatTemplateResult,
    pub prompt_token_count: usize,
    pub effective_ctx: usize,
    /// Wall-clock time for template application + tokenization + prompt
    /// prefill decode.
    pub prefill_ms: u64,
    /// Leading prompt tokens served from the retained KV cache.
    pub reused_prefix_tokens: usize,
    /// Context owning this generation's KV cache when the retained session was
    /// deliberately left intact. Dropped once the generation drains. `None`
    /// means the generation runs on `GenerationContext::session`.
    pub transient: Option<SessionKv>,
}

/// How the prompt for an incoming request relates to a retained KV cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PrefillPlan {
    /// No usable retained context; build one and decode the whole prompt.
    CreateContext,
    /// The retained context fits but its contents do not; clear the cache and
    /// decode the whole prompt into it.
    FullPrefillInPlace,
    /// The retained cache shares this many leading tokens with the prompt.
    ReusePrefix(usize),
    /// The prompt shares nothing useful with a much larger retained cache. Run
    /// in a throwaway context and leave the retained cache intact.
    SacrificialContext,
}

pub(super) fn common_prefix_len(cached: &[LlamaToken], prompt: &[LlamaToken]) -> usize {
    cached
        .iter()
        .zip(prompt)
        .take_while(|(cached, prompt)| cached == prompt)
        .count()
}

/// Whether a non-matching prompt is small enough that the retained cache is
/// worth more than this generation's convenience.
///
/// Retention belongs to the most expensive prefix. A prompt at most half the
/// size of the cache prefills in a fraction of the cache's rebuild cost and
/// gains nothing from being retained itself, so it runs somewhere else rather
/// than evicting a prefix that costs seconds to reconstruct. Hosts routinely
/// interleave exactly this shape: a short side call (memory extraction,
/// summarisation) between long conversational turns.
fn is_sacrificial_prompt(prompt_tokens: usize, cached_tokens: usize) -> bool {
    prompt_tokens > 0 && prompt_tokens.saturating_mul(2) < cached_tokens
}

/// Decide how to prefill `prompt` given the retained cache's `(tokens, n_ctx)`.
pub(super) fn prefill_plan(
    retained: Option<(&[LlamaToken], u32)>,
    prompt: &[LlamaToken],
    effective_ctx: u32,
) -> PrefillPlan {
    let Some((cached, cached_n_ctx)) = retained else {
        return PrefillPlan::CreateContext;
    };
    if cached_n_ctx != effective_ctx {
        return PrefillPlan::CreateContext;
    }
    // llama.cpp needs a non-empty batch to produce logits, so at least the
    // final prompt token must always be decoded.
    let reusable = common_prefix_len(cached, prompt).min(prompt.len().saturating_sub(1));
    if reusable >= REUSE_MIN_TOKENS {
        return PrefillPlan::ReusePrefix(reusable);
    }
    if is_sacrificial_prompt(prompt.len(), cached.len()) {
        return PrefillPlan::SacrificialContext;
    }
    PrefillPlan::FullPrefillInPlace
}

/// Output budget granted to a sacrificial generation when the model settings do
/// not cap it.
///
/// A throwaway context is sized to the prompt plus this budget rather than to
/// the full window, because it is resident *alongside* the retained cache and
/// llama.cpp commits KV memory for the whole `n_ctx` up front. On an 8 GB Jetson
/// two full 4096-token caches is the difference between fitting and an NvMap
/// allocation failure. The trade is that a sacrificial turn cannot generate more
/// than this many tokens, well above what the short side calls this path exists
/// for ever produce.
const SACRIFICIAL_OUTPUT_TOKENS: usize = 2048;

/// Window for a throwaway context: enough for the prompt and a full output
/// budget, never more than the request's own effective context.
fn sacrificial_context_size(
    prompt_tokens: usize,
    max_output_tokens: Option<usize>,
    effective_ctx: u32,
) -> u32 {
    let budget = max_output_tokens.unwrap_or(SACRIFICIAL_OUTPUT_TOKENS);
    let wanted = prompt_tokens.saturating_add(budget);
    u32::try_from(wanted).unwrap_or(u32::MAX).min(effective_ctx)
}

pub(super) struct StopSuffixTrimmer {
    pending: String,
    stops: Vec<String>,
}

impl StopSuffixTrimmer {
    pub(super) fn new(stops: &[String]) -> Self {
        Self {
            pending: String::new(),
            stops: stops
                .iter()
                .filter(|stop| !stop.is_empty())
                .cloned()
                .collect(),
        }
    }

    pub(super) fn push(&mut self, chunk: &str) -> (String, bool) {
        if self.stops.is_empty() {
            return (chunk.to_string(), false);
        }

        self.pending.push_str(chunk);

        if let Some(stop) = self
            .stops
            .iter()
            .filter(|stop| self.pending.ends_with(stop.as_str()))
            .max_by_key(|stop| stop.len())
        {
            let emit_len = self.pending.len() - stop.len();
            let _stop = self.pending.split_off(emit_len);
            let emit = std::mem::take(&mut self.pending);
            return (emit, true);
        }

        let hold_len = self
            .pending
            .char_indices()
            .map(|(idx, _)| idx)
            .chain(std::iter::once(self.pending.len()))
            .filter(|idx| {
                self.pending
                    .get(*idx..)
                    .is_some_and(|suffix| self.stops.iter().any(|stop| stop.starts_with(suffix)))
            })
            .map(|idx| self.pending.len() - idx)
            .max()
            .unwrap_or(0);

        let emit_len = self.pending.len() - hold_len;
        let keep = self.pending.split_off(emit_len);
        let emit = std::mem::replace(&mut self.pending, keep);
        (emit, false)
    }

    pub(super) fn finish(&mut self) -> String {
        std::mem::take(&mut self.pending)
    }
}

/// Estimate the maximum context length that can fit in available accelerator/CPU
/// memory based on the model's KV cache requirements.
///
/// `retained_kv_tokens` is the window size of a KV cache already held for this
/// model. That memory is released before any replacement is allocated, so it
/// counts towards what this request can spend; without it the estimate would
/// shrink on every turn that keeps a cache alive.
///
/// Returns `None` if the model architecture values are unavailable.
pub(super) fn estimate_max_context_for_memory(
    model: &LlamaModel,
    backend: &LlamaCppBackend,
    mmproj_overhead_bytes: u64,
    retained_kv_tokens: usize,
) -> Option<usize> {
    let raw_available = backend.available_memory_bytes();
    if raw_available == 0 {
        return None;
    }

    let n_layer = model.n_layer() as u64;
    let n_head_kv = model.n_head_kv() as u64;
    let n_head = model.n_head() as u64;
    let n_embd = model.n_embd() as u64;

    if n_head == 0 || n_layer == 0 || n_head_kv == 0 || n_embd == 0 {
        return None;
    }

    // For MLA (Multi-head Latent Attention) models like DeepSeek/GLM, the actual KV cache
    // dimensions differ from n_head_kv * head_dim. Read the true dimensions from GGUF metadata.
    let arch = model
        .meta_val_str("general.architecture")
        .unwrap_or_default();
    let head_dim = n_embd / n_head;
    let k_per_head = model
        .meta_val_str(&format!("{arch}.attention.key_length"))
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(head_dim);
    let v_per_head = model
        .meta_val_str(&format!("{arch}.attention.value_length"))
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(head_dim);

    // Total KV dimensions across all KV heads, times n_layer, times 2 bytes (f16) per element
    let bytes_per_token = (k_per_head + v_per_head) * n_head_kv * n_layer * 2;

    if bytes_per_token == 0 {
        return None;
    }

    let available = raw_available
        .saturating_add(retained_kv_tokens as u64 * bytes_per_token)
        .saturating_sub(mmproj_overhead_bytes);

    // Reserve memory for computation scratch buffers (attention, etc.) and other overhead.
    // The compute buffer can be 40-50% of the KV cache size for large models, so we
    // conservatively use only half the available memory for the KV cache.
    let usable = (available as f64 * 0.5) as u64;

    Some((usable / bytes_per_token) as usize)
}

pub(super) fn context_cap(
    settings: &crate::local_model_registry::ModelSettings,
    context_limit: usize,
    n_ctx_train: usize,
    memory_max_ctx: Option<usize>,
) -> usize {
    // 1. Explicit context_size in model settings (highest priority)
    if let Some(ctx_size) = settings.context_size {
        return ctx_size as usize;
    }

    // 2. context_limit from registry or caller
    if context_limit > 0 {
        return context_limit.min(n_ctx_train);
    }

    // 3. GOOSE_CONTEXT_LIMIT env var — read directly as a host override.
    //    When the host application (e.g. GIAP) sets this, it knows the
    //    platform's memory budget and accepts the KV cache cost.
    if let Ok(env_limit) = std::env::var("GOOSE_CONTEXT_LIMIT") {
        if let Ok(limit) = env_limit.parse::<usize>() {
            if limit > 0 {
                let capped = limit.min(n_ctx_train);
                tracing::info!(
                    "Using GOOSE_CONTEXT_LIMIT={} (host override, memory estimate skipped)",
                    capped,
                );
                return capped;
            }
        }
    }

    // 3. Fall back to memory-based estimation
    match memory_max_ctx {
        Some(mem_max) if mem_max < n_ctx_train => {
            tracing::info!(
                "Capping context from {} to {} based on available memory",
                n_ctx_train,
                mem_max,
            );
            mem_max
        }
        _ => n_ctx_train,
    }
}

pub(super) fn effective_context_size(
    prompt_token_count: usize,
    settings: &crate::local_model_registry::ModelSettings,
    context_limit: usize,
    n_ctx_train: usize,
    memory_max_ctx: Option<usize>,
) -> usize {
    let limit = context_cap(settings, context_limit, n_ctx_train, memory_max_ctx);
    let min_generation_headroom = 512;
    if prompt_token_count + min_generation_headroom > limit {
        tracing::warn!(
            "Prompt ({} tokens) + minimum headroom ({}) exceeds context limit ({})",
            prompt_token_count,
            min_generation_headroom,
            limit,
        );
    }
    limit
}

pub(super) fn build_context_params(
    ctx_size: u32,
    settings: &crate::local_model_registry::ModelSettings,
) -> LlamaContextParams {
    let mut params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(ctx_size));

    if let Some(n_batch) = settings.n_batch {
        params = params.with_n_batch(n_batch);
    }
    if let Some(n_threads) = settings.n_threads {
        params = params.with_n_threads(n_threads);
        params = params.with_n_threads_batch(n_threads);
    }
    if let Some(flash_attn) = settings.flash_attention {
        let policy = if flash_attn { 1 } else { 0 };
        params = params.with_flash_attention_policy(policy);
    }

    params
}

pub(super) fn build_sampler(settings: &crate::local_model_registry::ModelSettings) -> LlamaSampler {
    use crate::local_model_registry::SamplingConfig;

    let has_penalties = settings.repeat_penalty != 1.0
        || settings.frequency_penalty != 0.0
        || settings.presence_penalty != 0.0;

    let mut samplers: Vec<LlamaSampler> = Vec::new();

    if has_penalties {
        samplers.push(LlamaSampler::penalties(
            settings.repeat_last_n,
            settings.repeat_penalty,
            settings.frequency_penalty,
            settings.presence_penalty,
        ));
    }

    match &settings.sampling {
        SamplingConfig::Greedy => {
            samplers.push(LlamaSampler::greedy());
        }
        SamplingConfig::Temperature {
            temperature,
            top_k,
            top_p,
            min_p,
            seed,
        } => {
            samplers.push(LlamaSampler::top_k(*top_k));
            samplers.push(LlamaSampler::top_p(*top_p, 1));
            samplers.push(LlamaSampler::min_p(*min_p, 1));
            samplers.push(LlamaSampler::temp(*temperature));
            samplers.push(LlamaSampler::dist(seed.unwrap_or(0)));
        }
        SamplingConfig::MirostatV2 { tau, eta, seed } => {
            samplers.push(LlamaSampler::mirostat_v2(seed.unwrap_or(0), *tau, *eta));
        }
    }

    if samplers.len() == 1 {
        samplers.pop().unwrap()
    } else {
        LlamaSampler::chain_simple(samplers)
    }
}

/// Validate prompt tokens against memory limits and compute the effective
/// context size. Returns `(prompt_token_count, effective_ctx)`.
pub(super) fn validate_and_compute_context(
    model: &LlamaModel,
    has_mtmd: bool,
    backend: &LlamaCppBackend,
    prompt_token_count: usize,
    context_limit: usize,
    settings: &crate::local_model_registry::ModelSettings,
    retained_kv_tokens: usize,
) -> Result<(usize, usize), ProviderError> {
    let n_ctx_train = model.n_ctx_train() as usize;
    let mmproj_overhead = if has_mtmd {
        settings.mmproj_size_bytes
    } else {
        0
    };
    let memory_max_ctx =
        estimate_max_context_for_memory(model, backend, mmproj_overhead, retained_kv_tokens);
    let effective_ctx = effective_context_size(
        prompt_token_count,
        settings,
        context_limit,
        n_ctx_train,
        memory_max_ctx,
    );
    if let Some(mem_max) = memory_max_ctx {
        if prompt_token_count > mem_max {
            return Err(ProviderError::ContextLengthExceeded(format!(
                "Prompt ({} tokens) exceeds estimated memory capacity ({} tokens). \
                 Try a smaller model or reduce conversation length.",
                prompt_token_count, mem_max,
            )));
        }
    }
    if prompt_token_count >= effective_ctx {
        return Err(ProviderError::ContextLengthExceeded(format!(
            "Prompt ({} tokens) exceeds context limit ({} tokens). \
             Try reducing conversation length.",
            prompt_token_count, effective_ctx,
        )));
    }
    Ok((prompt_token_count, effective_ctx))
}

/// Decode `tokens` into sequence 0 starting at absolute position `start_pos`.
///
/// Positions are explicit so the caller controls where the batch lands relative
/// to whatever the KV cache already holds; llama.cpp requires them to continue
/// exactly from the sequence's highest cached position. The last token of each
/// chunk requests logits, matching `llama_batch_get_one`, so only the position
/// assignment differs from an implicitly positioned batch.
fn decode_tokens(
    ctx: &mut LlamaContext<'_>,
    tokens: &[LlamaToken],
    start_pos: usize,
) -> Result<(), ProviderError> {
    if tokens.is_empty() {
        return Ok(());
    }

    let n_batch = (ctx.n_batch() as usize).max(1);
    let mut batch = LlamaBatch::new(n_batch.min(tokens.len()), 1);

    for (chunk_index, chunk) in tokens.chunks(n_batch).enumerate() {
        batch.clear();
        let chunk_last = chunk.len() - 1;
        for (offset, token) in chunk.iter().enumerate() {
            let pos = i32::try_from(start_pos + chunk_index * n_batch + offset).map_err(|_| {
                ProviderError::ExecutionError("Prompt position exceeds i32 range".to_string())
            })?;
            batch
                .add(*token, pos, &[0], offset == chunk_last)
                .map_err(|e| {
                    ProviderError::ExecutionError(format!("Failed to build batch: {e}"))
                })?;
        }
        ctx.decode(&mut batch)
            .map_err(|e| ProviderError::ExecutionError(format!("Prefill decode failed: {e}")))?;
    }

    Ok(())
}

/// Why a partial KV cache reuse could not be completed.
enum PrefixReuseError {
    /// llama.cpp declined the partial removal; the cache is untouched.
    Refused,
    /// The suffix decode failed; the cache contents can no longer be trusted.
    Poisoned(ProviderError),
}

fn reuse_prefix(
    kv: &mut SessionKv,
    prompt: &[LlamaToken],
    reusable: usize,
) -> Result<(), PrefixReuseError> {
    let p0 = u32::try_from(reusable).map_err(|_| PrefixReuseError::Refused)?;
    match kv.ctx.clear_kv_cache_seq(Some(0), Some(p0), None) {
        Ok(true) => {}
        Ok(false) | Err(_) => return Err(PrefixReuseError::Refused),
    }

    kv.tokens.truncate(reusable);
    decode_tokens(&mut kv.ctx, &prompt[reusable..], reusable)
        .map_err(PrefixReuseError::Poisoned)?;
    kv.tokens.extend_from_slice(&prompt[reusable..]);
    Ok(())
}

/// Decode the whole prompt from position 0, reusing the retained context when
/// its window matches and building a new one otherwise.
fn full_prefill(
    session: &mut Option<SessionKv>,
    model: &Arc<LlamaModel>,
    backend: &LlamaCppBackend,
    settings: &ModelSettings,
    prompt: &[LlamaToken],
    n_ctx: u32,
) -> Result<(), ProviderError> {
    if session.as_ref().is_none_or(|kv| kv.n_ctx != n_ctx) {
        // Release the old context before allocating its replacement so both
        // KV caches are never resident at once.
        *session = None;
        *session = Some(SessionKv::create(model, backend, n_ctx, settings)?);
    }

    let kv = session
        .as_mut()
        .expect("session was just populated or already matched n_ctx");
    kv.tokens.clear();
    kv.ctx.clear_kv_cache();
    match decode_tokens(&mut kv.ctx, prompt, 0) {
        Ok(()) => {
            kv.tokens.extend_from_slice(prompt);
            Ok(())
        }
        Err(e) => {
            *session = None;
            Err(e)
        }
    }
}

/// Outcome of preparing a KV cache for one generation.
struct PrefilledPrompt {
    /// Leading prompt tokens served from a cache rather than decoded.
    reused_prefix_tokens: usize,
    /// Throwaway context owning this generation's KV cache. `None` means the
    /// generation runs on the retained session.
    transient: Option<SessionKv>,
    /// Window the generation must respect: the throwaway context's own, smaller
    /// window on a sacrificial turn, otherwise the request's effective context.
    effective_ctx: usize,
}

/// Prepare a KV cache holding `prompt`, with logits available for its last
/// token.
fn prefill_prompt(
    session: &mut Option<SessionKv>,
    model: &Arc<LlamaModel>,
    backend: &LlamaCppBackend,
    settings: &ModelSettings,
    prompt: &[LlamaToken],
    effective_ctx: usize,
) -> Result<PrefilledPrompt, ProviderError> {
    let n_ctx = u32::try_from(effective_ctx).map_err(|_| {
        ProviderError::ExecutionError(format!("Context size {effective_ctx} exceeds u32 range"))
    })?;

    let plan = prefill_plan(
        session.as_ref().map(|kv| (kv.tokens.as_slice(), kv.n_ctx)),
        prompt,
        n_ctx,
    );
    tracing::debug!(
        ?plan,
        cached_tokens = session.as_ref().map(|kv| kv.tokens.len()),
        cached_n_ctx = session.as_ref().map(|kv| kv.n_ctx),
        prompt_tokens = prompt.len(),
        n_ctx,
        "prompt prefill plan"
    );

    match plan {
        PrefillPlan::ReusePrefix(reusable) => {
            let kv = session
                .as_mut()
                .expect("ReusePrefix is only produced when a session is retained");
            match reuse_prefix(kv, prompt, reusable) {
                Ok(()) => {
                    return Ok(PrefilledPrompt {
                        reused_prefix_tokens: reusable,
                        transient: None,
                        effective_ctx,
                    })
                }
                Err(PrefixReuseError::Refused) => {
                    tracing::debug!(
                        reusable,
                        "llama.cpp declined partial KV removal; prefilling the whole prompt"
                    );
                }
                Err(PrefixReuseError::Poisoned(e)) => {
                    tracing::warn!(error = %e, "suffix decode failed; rebuilding the generation context");
                    *session = None;
                }
            }
        }
        PrefillPlan::SacrificialContext => {
            let transient_n_ctx =
                sacrificial_context_size(prompt.len(), settings.max_output_tokens, n_ctx);
            match sacrificial_prefill(model, backend, settings, prompt, transient_n_ctx) {
                Ok(kv) => {
                    return Ok(PrefilledPrompt {
                        reused_prefix_tokens: 0,
                        transient: Some(kv),
                        effective_ctx: transient_n_ctx as usize,
                    })
                }
                Err(e) => {
                    // Keeping the turn alive matters more than keeping the
                    // cache; fall through and prefill in place.
                    tracing::warn!(error = %e, "throwaway context unavailable; prefilling over the retained cache");
                }
            }
        }
        PrefillPlan::CreateContext | PrefillPlan::FullPrefillInPlace => {}
    }

    let retained = session.is_some();
    match full_prefill(session, model, backend, settings, prompt, n_ctx) {
        Ok(()) => {}
        Err(e) if retained => {
            // `full_prefill` dropped the poisoned context, so this attempt runs
            // against a freshly created one.
            tracing::warn!(error = %e, "prompt prefill failed; retrying with a new context");
            full_prefill(session, model, backend, settings, prompt, n_ctx)?;
        }
        Err(e) => return Err(e),
    }
    Ok(PrefilledPrompt {
        reused_prefix_tokens: 0,
        transient: None,
        effective_ctx,
    })
}

/// Build a context that lives only for this generation and decode the whole
/// prompt into it, leaving any retained cache untouched.
fn sacrificial_prefill(
    model: &Arc<LlamaModel>,
    backend: &LlamaCppBackend,
    settings: &ModelSettings,
    prompt: &[LlamaToken],
    n_ctx: u32,
) -> Result<SessionKv, ProviderError> {
    let mut kv = SessionKv::create(model, backend, n_ctx, settings)?;
    decode_tokens(&mut kv.ctx, prompt, 0)?;
    Ok(kv)
}

/// Tokenize text + images via mtmd into a freshly built context.
///
/// Image embeddings are not a plain token prefix, so the resulting session is
/// stored with an empty token list: the next text-only request falls through to
/// a full prefill instead of attempting reuse.
fn prefill_multimodal(
    model: &Arc<LlamaModel>,
    mtmd_ctx: Option<&MtmdContext>,
    backend: &LlamaCppBackend,
    prompt_text: &str,
    images: &[ExtractedImage],
    context_limit: usize,
    settings: &ModelSettings,
) -> Result<(SessionKv, usize, usize), ProviderError> {
    let mtmd_ctx = mtmd_ctx.ok_or_else(|| {
        ProviderError::ExecutionError(
            "This model does not have vision support. Download the vision encoder from \
             Settings > Local Inference, or use a text-only message."
                .to_string(),
        )
    })?;

    let bitmaps: Vec<MtmdBitmap> = images
        .iter()
        .map(|img| {
            MtmdBitmap::from_buffer(mtmd_ctx, &img.bytes)
                .map_err(|e| ProviderError::ExecutionError(format!("Failed to decode image: {e}")))
        })
        .collect::<Result<_, _>>()?;

    let bitmap_refs: Vec<&MtmdBitmap> = bitmaps.iter().collect();

    let input_text = MtmdInputText {
        text: prompt_text.to_string(),
        add_special: true,
        parse_special: true,
    };
    let chunks = mtmd_ctx.tokenize(input_text, &bitmap_refs).map_err(|e| {
        ProviderError::ExecutionError(format!("Multimodal tokenization failed: {e}"))
    })?;

    let prompt_token_count = chunks.total_tokens();

    let n_ctx_train = model.n_ctx_train() as usize;
    let mmproj_overhead = settings.mmproj_size_bytes;
    let memory_max_ctx = estimate_max_context_for_memory(model, backend, mmproj_overhead, 0);
    let effective_ctx = effective_context_size(
        prompt_token_count,
        settings,
        context_limit,
        n_ctx_train,
        memory_max_ctx,
    );

    let min_generation_headroom = 512;
    if prompt_token_count + min_generation_headroom > effective_ctx {
        return Err(ProviderError::ContextLengthExceeded(format!(
            "Multimodal prompt ({prompt_token_count} tokens including images) exceeds \
             context limit ({effective_ctx} tokens)",
        )));
    }

    let n_ctx = u32::try_from(effective_ctx).map_err(|_| {
        ProviderError::ExecutionError(format!("Context size {effective_ctx} exceeds u32 range"))
    })?;
    let session = SessionKv::create(model, backend, n_ctx, settings)?;

    let n_batch = session.ctx.n_batch() as i32;
    let _n_past = chunks
        .eval_chunks(mtmd_ctx, &session.ctx, 0, 0, n_batch, true)
        .map_err(|e| ProviderError::ExecutionError(format!("Multimodal eval failed: {e}")))?;

    Ok((session, prompt_token_count, effective_ctx))
}

pub(super) fn prepare_generation(
    ctx: &mut GenerationContext<'_>,
    oai_messages_json: &str,
    full_tools_json: Option<&str>,
    compact_tools_json: Option<&str>,
) -> Result<PreparedGeneration, ProviderError> {
    let prefill_started = std::time::Instant::now();
    let apply_template = |tools: Option<&str>| {
        let params = OpenAIChatTemplateParams {
            messages_json: oai_messages_json,
            tools_json: tools,
            tool_choice: None,
            json_schema: None,
            grammar: None,
            reasoning_format: if ctx.settings.enable_thinking {
                Some("auto")
            } else {
                None
            },
            chat_template_kwargs: None,
            add_generation_prompt: true,
            use_jinja: true,
            parallel_tool_calls: false,
            enable_thinking: ctx.settings.enable_thinking,
            add_bos: false,
            add_eos: false,
            parse_tool_calls: true,
        };
        ctx.model
            .apply_chat_template_oaicompat(ctx.template, &params)
    };

    let min_generation_headroom = 512;
    let n_ctx_train = ctx.model.n_ctx_train() as usize;
    let mmproj_overhead = if ctx.mtmd_ctx.is_some() {
        ctx.settings.mmproj_size_bytes
    } else {
        0
    };
    let retained_kv_tokens = retained_kv_tokens(ctx.session);
    let memory_max_ctx = estimate_max_context_for_memory(
        ctx.model,
        ctx.backend,
        mmproj_overhead,
        retained_kv_tokens,
    );
    let cap = context_cap(ctx.settings, ctx.context_limit, n_ctx_train, memory_max_ctx);
    let token_budget = cap.saturating_sub(min_generation_headroom);
    let estimated_image_tokens = ctx.images.len() * ctx.settings.image_token_estimate;

    let template_result = match apply_template(full_tools_json) {
        Ok(r) => {
            let token_count = ctx
                .model
                .str_to_token(&r.prompt, AddBos::Never)
                .map(|t| t.len())
                .unwrap_or(0);
            if token_count + estimated_image_tokens > token_budget {
                apply_template(compact_tools_json).unwrap_or(r)
            } else {
                r
            }
        }
        Err(e) => {
            tracing::warn!(
                error = %e,
                "Failed to apply llama.cpp OpenAI-compatible chat template"
            );
            match apply_template(compact_tools_json) {
                Ok(r) => r,
                Err(compact_err) => {
                    return Err(ProviderError::ExecutionError(format!(
                        "Failed to apply chat template with llama.cpp's Jinja renderer. This usually means the selected built-in template name does not exist, the embedded or custom template is invalid, or the template is incompatible with the current message shape. Select a valid llama.cpp built-in template name, configure a custom inline Jinja template, or use a GGUF with valid tokenizer.chat_template metadata. Full tools error: {e}; compact tools error: {compact_err}"
                    )));
                }
            }
        }
    };

    let _ = ctx.log.write(
        &serde_json::json!({"applied_prompt": &template_result.prompt}),
        None,
    );

    let (prompt_token_count, prefilled) = if !ctx.images.is_empty() {
        // A prompt carrying image embeddings never reuses a retained cache:
        // release it first so the two KV caches are not resident together.
        *ctx.session = None;
        let (session, ptc, ectx) = prefill_multimodal(
            ctx.model,
            ctx.mtmd_ctx,
            ctx.backend,
            &template_result.prompt,
            ctx.images,
            ctx.context_limit,
            ctx.settings,
        )?;
        *ctx.session = Some(session);
        (
            ptc,
            PrefilledPrompt {
                reused_prefix_tokens: 0,
                transient: None,
                effective_ctx: ectx,
            },
        )
    } else {
        let tokens = ctx
            .model
            .str_to_token(&template_result.prompt, AddBos::Never)
            .map_err(|e| ProviderError::ExecutionError(e.to_string()))?;
        let (ptc, ectx) = validate_and_compute_context(
            ctx.model,
            ctx.mtmd_ctx.is_some(),
            ctx.backend,
            tokens.len(),
            ctx.context_limit,
            ctx.settings,
            retained_kv_tokens,
        )?;
        let prefilled = prefill_prompt(
            ctx.session,
            ctx.model,
            ctx.backend,
            ctx.settings,
            &tokens,
            ectx,
        )?;
        (ptc, prefilled)
    };

    Ok(PreparedGeneration {
        template_result,
        prompt_token_count,
        effective_ctx: prefilled.effective_ctx,
        prefill_ms: prefill_started.elapsed().as_millis() as u64,
        reused_prefix_tokens: prefilled.reused_prefix_tokens,
        transient: prefilled.transient,
    })
}

/// Action to take after processing a generated token piece.
pub(super) enum TokenAction {
    Continue,
    Stop,
}

/// Run the autoregressive generation loop. Calls `on_piece` for each non-empty
/// token piece. The callback returns `TokenAction::Stop` to break early.
/// Returns the total number of generated tokens, or `ContextLengthExceeded`
/// if the model exhausted the available context window.
///
/// Every token appended to the KV cache is pushed onto `decoded`, so the caller
/// can extend a retained token sequence with exactly what the cache now holds.
/// The token a stop condition fires on is sampled but never decoded, and is
/// therefore absent from `decoded`.
pub(super) fn generation_loop(
    model: &LlamaModel,
    ctx: &mut LlamaContext<'_>,
    settings: &crate::local_model_registry::ModelSettings,
    prompt_token_count: usize,
    effective_ctx: usize,
    decoded: &mut Vec<LlamaToken>,
    mut on_piece: impl FnMut(&str) -> Result<TokenAction, ProviderError>,
) -> Result<i32, ProviderError> {
    let mut sampler = build_sampler(settings);
    let context_headroom = effective_ctx.saturating_sub(prompt_token_count);
    let max_output = if let Some(max) = settings.max_output_tokens {
        context_headroom.min(max)
    } else {
        context_headroom
    };
    let hit_context_limit = settings
        .max_output_tokens
        .is_none_or(|max| context_headroom <= max);
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut output_token_count: i32 = 0;
    let mut exhausted_loop = true;

    for _ in 0..max_output {
        let token = sampler.sample(ctx, -1);
        sampler.accept(token);

        if model.is_eog_token(token) {
            exhausted_loop = false;
            break;
        }

        output_token_count += 1;

        let piece = model
            .token_to_piece(token, &mut decoder, true, None)
            .map_err(|e| ProviderError::ExecutionError(format!("Failed to decode token: {}", e)))?;

        if !piece.is_empty() && matches!(on_piece(&piece)?, TokenAction::Stop) {
            exhausted_loop = false;
            break;
        }

        let next_tokens = [token];
        let mut next_batch = LlamaBatch::get_one(&next_tokens)
            .map_err(|e| ProviderError::ExecutionError(format!("Failed to create batch: {}", e)))?;
        ctx.decode(&mut next_batch)
            .map_err(|e| ProviderError::ExecutionError(format!("Decode failed: {}", e)))?;
        decoded.push(token);
    }

    if exhausted_loop && hit_context_limit {
        return Err(ProviderError::ContextLengthExceeded(format!(
            "Generation exhausted context window ({} prompt + {} generated = {} of {} limit)",
            prompt_token_count,
            output_token_count,
            prompt_token_count as i32 + output_token_count,
            effective_ctx,
        )));
    }

    Ok(output_token_count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::local_model_registry::ModelSettings;

    fn default_settings() -> ModelSettings {
        ModelSettings::default()
    }

    #[test]
    fn test_effective_context_size_uses_full_limit() {
        assert_eq!(
            effective_context_size(100, &default_settings(), 4096, 4096, None),
            4096
        );
    }

    #[test]
    fn test_effective_context_size_capped_by_limit() {
        assert_eq!(
            effective_context_size(100, &default_settings(), 1024, 8192, None),
            1024
        );
    }

    #[test]
    fn test_effective_context_size_capped_by_memory() {
        assert_eq!(
            effective_context_size(100, &default_settings(), 4096, 4096, Some(800)),
            800
        );
    }

    #[test]
    fn test_effective_context_size_memory_smaller_than_prompt() {
        assert_eq!(
            effective_context_size(600, &default_settings(), 4096, 4096, Some(700)),
            700
        );
    }

    #[test]
    fn test_effective_context_size_zero_limit_uses_train() {
        assert_eq!(
            effective_context_size(100, &default_settings(), 0, 2048, None),
            2048
        );
    }

    #[test]
    fn test_effective_context_size_prompt_exceeds_all_limits() {
        assert_eq!(
            effective_context_size(5000, &default_settings(), 4096, 4096, None),
            4096
        );
    }

    #[test]
    fn test_context_cap_with_settings_override() {
        let mut settings = default_settings();
        settings.context_size = Some(2048);
        assert_eq!(context_cap(&settings, 4096, 8192, Some(1024)), 2048);
    }

    #[test]
    fn test_context_cap_without_override() {
        assert_eq!(context_cap(&default_settings(), 4096, 8192, None), 4096);
    }

    #[test]
    fn test_context_cap_memory_limited() {
        assert_eq!(
            context_cap(&default_settings(), 4096, 8192, Some(2048)),
            2048
        );
    }

    fn tokens(ids: impl IntoIterator<Item = i32>) -> Vec<LlamaToken> {
        ids.into_iter().map(LlamaToken::new).collect()
    }

    /// A prompt whose first `shared` tokens match `cached`, then diverges.
    fn diverging_prompt(shared: usize, extra: usize) -> Vec<LlamaToken> {
        tokens((0..shared as i32).chain((0..extra as i32).map(|i| 100_000 + i)))
    }

    #[test]
    fn common_prefix_len_counts_shared_leading_tokens() {
        assert_eq!(
            common_prefix_len(&tokens([1, 2, 3, 4]), &tokens([1, 2, 9, 4])),
            2
        );
        assert_eq!(common_prefix_len(&tokens([1, 2]), &tokens([1, 2, 3])), 2);
        assert_eq!(common_prefix_len(&tokens([1, 2, 3]), &tokens([1, 2])), 2);
        assert_eq!(common_prefix_len(&tokens([9]), &tokens([1, 2])), 0);
        assert_eq!(common_prefix_len(&[], &tokens([1, 2])), 0);
        assert_eq!(common_prefix_len(&tokens([1, 2]), &[]), 0);
    }

    #[test]
    fn prefill_plan_without_retained_session_creates_context() {
        assert_eq!(
            prefill_plan(None, &diverging_prompt(1000, 8), 4096),
            PrefillPlan::CreateContext
        );
    }

    #[test]
    fn prefill_plan_on_context_size_change_creates_context() {
        let cached = tokens(0..1000);
        let prompt = diverging_prompt(1000, 8);
        assert_eq!(
            prefill_plan(Some((&cached, 2048)), &prompt, 4096),
            PrefillPlan::CreateContext
        );
    }

    #[test]
    fn prefill_plan_reuses_long_shared_prefix() {
        let cached = tokens(0..1000);
        let prompt = diverging_prompt(900, 40);
        assert_eq!(
            prefill_plan(Some((&cached, 4096)), &prompt, 4096),
            PrefillPlan::ReusePrefix(900)
        );
    }

    #[test]
    fn prefill_plan_rejects_short_shared_prefix() {
        // Comparable in size to the cache, so the sacrificial path does not
        // apply and a genuinely new conversation shape takes the context over.
        let cached = tokens(0..1000);
        let prompt = diverging_prompt(REUSE_MIN_TOKENS - 1, 745);
        assert_eq!(prompt.len(), 1000);
        assert_eq!(
            prefill_plan(Some((&cached, 4096)), &prompt, 4096),
            PrefillPlan::FullPrefillInPlace
        );
    }

    #[test]
    fn prefill_plan_reuses_at_exactly_the_threshold() {
        let cached = tokens(0..1000);
        let prompt = diverging_prompt(REUSE_MIN_TOKENS, 40);
        assert_eq!(
            prefill_plan(Some((&cached, 4096)), &prompt, 4096),
            PrefillPlan::ReusePrefix(REUSE_MIN_TOKENS)
        );
    }

    #[test]
    fn prefill_plan_always_leaves_a_token_to_decode() {
        let cached = tokens(0..1000);
        let prompt = tokens(0..1000);
        assert_eq!(
            prefill_plan(Some((&cached, 4096)), &prompt, 4096),
            PrefillPlan::ReusePrefix(999)
        );
    }

    #[test]
    fn prefill_plan_reuses_when_prompt_extends_the_cache() {
        let cached = tokens(0..1000);
        let prompt = tokens(0..1400);
        assert_eq!(
            prefill_plan(Some((&cached, 4096)), &prompt, 4096),
            PrefillPlan::ReusePrefix(1000)
        );
    }

    #[test]
    fn sacrificial_boundary_is_exactly_half_the_cache() {
        // prompt * 2 < cached
        assert!(is_sacrificial_prompt(499, 1000));
        // prompt * 2 == cached — not smaller, so no sacrifice
        assert!(!is_sacrificial_prompt(500, 1000));
        assert!(!is_sacrificial_prompt(501, 1000));
        assert!(!is_sacrificial_prompt(1000, 1000));
        assert!(!is_sacrificial_prompt(7157, 772));
        // No cache to protect.
        assert!(!is_sacrificial_prompt(0, 0));
        assert!(!is_sacrificial_prompt(100, 0));
        // A degenerate empty prompt has nothing to decode elsewhere.
        assert!(!is_sacrificial_prompt(0, 1000));
    }

    #[test]
    fn prefill_plan_sacrifices_a_small_non_matching_prompt() {
        // The observed interleave: a 378-token side call against a 7138-token
        // conversational cache must not evict it.
        let cached = tokens(0..7138);
        let prompt = diverging_prompt(8, 370);
        assert_eq!(
            prefill_plan(Some((&cached, 4096)), &prompt, 4096),
            PrefillPlan::SacrificialContext
        );
    }

    #[test]
    fn prefill_plan_just_below_the_sacrificial_threshold() {
        let cached = tokens(0..1000);
        // 499 tokens, sharing fewer than REUSE_MIN: 499 * 2 < 1000.
        let prompt = diverging_prompt(8, 491);
        assert_eq!(prompt.len(), 499);
        assert_eq!(
            prefill_plan(Some((&cached, 4096)), &prompt, 4096),
            PrefillPlan::SacrificialContext
        );
    }

    #[test]
    fn prefill_plan_just_above_the_sacrificial_threshold_prefills_in_place() {
        let cached = tokens(0..1000);
        // 500 tokens: 500 * 2 == 1000, so the cache is not worth protecting.
        let prompt = diverging_prompt(8, 492);
        assert_eq!(prompt.len(), 500);
        assert_eq!(
            prefill_plan(Some((&cached, 4096)), &prompt, 4096),
            PrefillPlan::FullPrefillInPlace
        );
    }

    #[test]
    fn reuse_wins_over_sacrifice_when_the_prefix_is_long_enough() {
        // A small prompt that nonetheless shares REUSE_MIN tokens with the cache
        // is cheaper to serve from it than to prefill elsewhere.
        let cached = tokens(0..7138);
        let prompt = diverging_prompt(REUSE_MIN_TOKENS, 40);
        assert!(is_sacrificial_prompt(prompt.len(), cached.len()));
        assert_eq!(
            prefill_plan(Some((&cached, 4096)), &prompt, 4096),
            PrefillPlan::ReusePrefix(REUSE_MIN_TOKENS)
        );
    }

    #[test]
    fn sacrifice_needs_a_retained_context_of_the_same_size() {
        let cached = tokens(0..7138);
        let prompt = diverging_prompt(8, 370);
        // A window change forces a rebuild regardless of prompt size.
        assert_eq!(
            prefill_plan(Some((&cached, 2048)), &prompt, 4096),
            PrefillPlan::CreateContext
        );
        assert_eq!(
            prefill_plan(None, &prompt, 4096),
            PrefillPlan::CreateContext
        );
    }

    #[test]
    fn sacrificial_context_is_sized_to_the_prompt_plus_an_output_budget() {
        assert_eq!(
            sacrificial_context_size(378, None, 4096),
            378 + SACRIFICIAL_OUTPUT_TOKENS as u32
        );
        // Never larger than the request's own window.
        assert_eq!(sacrificial_context_size(378, None, 1024), 1024);
        // An explicit output cap is respected instead of the default budget.
        assert_eq!(sacrificial_context_size(378, Some(256), 4096), 634);
        // Saturating arithmetic, not a panic.
        assert_eq!(sacrificial_context_size(usize::MAX, None, 4096), 4096);
    }

    #[test]
    fn sacrificial_context_stays_smaller_than_the_retained_window() {
        // Sacrifice requires prompt * 2 < cached <= n_ctx, so any prompt that
        // reaches this path is under half the window. At the Jetson's 4096 that
        // bounds the throwaway cache strictly below a second full one.
        let n_ctx: u32 = 4096;
        for prompt in [1usize, 378, 1000, 2047] {
            assert!(is_sacrificial_prompt(prompt, n_ctx as usize));
            let transient = sacrificial_context_size(prompt, None, n_ctx);
            assert!(
                transient < n_ctx,
                "prompt {prompt} produced a throwaway window of {transient}, not below {n_ctx}"
            );
        }
    }

    #[test]
    fn prefill_plan_with_empty_cached_tokens_prefills_in_place() {
        assert_eq!(
            prefill_plan(Some((&[], 4096)), &diverging_prompt(1000, 8), 4096),
            PrefillPlan::FullPrefillInPlace
        );
    }

    #[test]
    fn prefill_plan_on_empty_prompt_prefills_in_place() {
        let cached = tokens(0..1000);
        assert_eq!(
            prefill_plan(Some((&cached, 4096)), &[], 4096),
            PrefillPlan::FullPrefillInPlace
        );
    }
}
