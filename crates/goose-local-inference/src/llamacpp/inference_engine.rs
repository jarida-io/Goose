use crate::backend::LocalInferenceBackend;
use crate::local_model_registry::ModelSettings;
use crate::multimodal::ExtractedImage;
use goose_provider_types::errors::ProviderError;
use goose_provider_types::request_log::{LoggerHandleExt, RequestLogHandle};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::{AddBos, ChatTemplateResult, LlamaChatTemplate, LlamaModel};
use llama_cpp_2::mtmd::{
    MtmdBitmap, MtmdContext, MtmdInputChunk, MtmdInputChunkType, MtmdInputChunks, MtmdInputText,
};
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

/// One span of a retained cache's media-bearing head, in decode order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum HeadChunk {
    /// Text tokens, one KV position each.
    Text(Vec<LlamaToken>),
    /// Media embeddings. Opaque here: what they contain is pinned by the source
    /// images `MediaHead` holds alongside, not by anything in this variant.
    Media { n_tokens: usize, n_pos: usize },
}

impl HeadChunk {
    fn n_pos(&self) -> usize {
        match self {
            Self::Text(tokens) => tokens.len(),
            Self::Media { n_pos, .. } => *n_pos,
        }
    }
}

/// Why an incoming prompt could not resume from a retained media head.
///
/// Carried rather than collapsed to a bool so the log says which input moved.
/// The causes have very different meanings for whoever is reading it: an image
/// change is the user sending a different picture, or the host replaying a
/// different number of them (both expected), whereas a chunk-shape change with
/// identical images is the *prompt around* the pictures having been rewritten.
///
/// The checks in [`MediaHead::mismatch`] run narrowest-cause first, so the
/// reported variant is the most specific one that explains the miss. A change
/// usually trips several at once — an extra image is also an extra chunk and
/// more positions. Order decides only which reason is logged: the head matches
/// if and only if every check agrees.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum HeadMismatch {
    /// A different number of images. This is the growing-image-list
    /// conversation, and the reason that shape gets no reuse.
    ImageCount,
    /// The same number of images, but different pixels in at least one.
    ImageBytes,
    /// Same images, different chunk list: the text around the pictures was
    /// rewritten. Two routine causes, both of which also move every media
    /// position — the chunk list is simply the narrower way to say it. The
    /// first is `prepare_generation` flipping between the full and compact tool
    /// schemas, which rewrites the whole rendered prompt including everything
    /// ahead of the first image. The second is any growth in the transcript
    /// ahead of the images. Either way no part of the head survives.
    ChunkShape,
    /// The head would occupy a different number of KV positions despite an
    /// identical chunk list. Not reachable from mtmd, whose position count is
    /// the sum over exactly those chunks on both sides. It is the tripwire for
    /// a `MediaHead` whose recorded `n_pos` disagrees with its own chunks,
    /// which would resume the decode at the wrong absolute position.
    Positions,
}

/// The leading part of a retained KV cache that carries media embeddings.
///
/// # Why the match is all-or-nothing
///
/// Media KV contents are not tokens, so they cannot be compared position by
/// position the way `SessionKv::tokens` is. The head is therefore matched
/// atomically: a prompt may resume from it only when it reproduces the head
/// exactly — the same chunk shape in the same order, built from the same image
/// bytes in the same order. That suffices because mtmd's tokenization and
/// preprocessing are deterministic functions of (prompt text, bitmaps), so
/// equal inputs place equal pixel batches at equal positions.
///
/// There is deliberately no partial head match. The head ends at the last media
/// chunk, so anything that changes it changes a chunk at or before an image, and
/// every image after that point moves to a different KV position. Matching a
/// prefix of the head would resume behind embeddings that no longer describe
/// what the prompt says they do. Degrading to no reuse is the correct outcome;
/// the cost is a rebuild, and the alternative is a wrong answer.
///
/// # What this actually buys, honestly
///
/// Only conversations whose image list is *already stable* turn by turn. The
/// four-turn production trace that motivated caching here (1 then 2 then 3
/// encodes per turn, prefill climbing 28s to 37s) is not one of them: its image
/// list grows every turn, so [`MediaHead::mismatch`] returns
/// [`HeadMismatch::ImageCount`] and reuse never fires on it. That trace is
/// addressed upstream of this engine, by the host capping how many historical
/// images it replays — not by this cache.
///
/// The shape this does serve is the one left over once that cap holds the list
/// steady: "here is a photo" followed by questions about that same photo. Turns
/// 2..N present a byte-identical image list, so they skip the vision encode and
/// the prefill of everything ahead of the tail.
///
/// Images promoted out of tool results (a camera look) never form a stable head
/// at all, and not only when the camera is looked at again. GIAP's provider shim
/// re-derives them from the newest frames every turn and appends them in a fresh
/// trailing carrier message, after the whole conversation. Two consequences:
/// the last image sits at the very end of the prompt, so the head is
/// effectively the entire prompt; and the transcript ahead of it grows each
/// turn, so the chunk list moves each turn. Even a session that looks once and
/// then asks follow-ups misses on [`HeadMismatch::ChunkShape`]. Only images
/// carried in the conversation itself, at a position the transcript no longer
/// changes, are reused here.
pub(super) struct MediaHead {
    chunks: Vec<HeadChunk>,
    /// Positions the head occupies: `[0, n_pos)`.
    n_pos: usize,
    /// Source bytes of every image in the request that built this head, in
    /// order. Compared byte for byte rather than hashed — a hash collision here
    /// would silently answer about a different picture, and images are small
    /// enough next to a resident model that the copy is not worth the risk.
    images: Vec<Vec<u8>>,
}

impl MediaHead {
    /// `None` when `(chunks, images)` reproduce this head exactly.
    ///
    /// Ordered narrowest cause first — see [`HeadMismatch`]. The byte compare
    /// runs ahead of the chunk compare so `ChunkShape` keeps the meaning its
    /// doc gives it ("same images, rewritten text"); that costs one memcmp of
    /// a few hundred KB on the rewritten-prompt path, against a rebuild
    /// measured in seconds.
    fn mismatch(
        &self,
        n_pos: usize,
        chunks: &[HeadChunk],
        images: &[ExtractedImage],
    ) -> Option<HeadMismatch> {
        if self.images.len() != images.len() {
            return Some(HeadMismatch::ImageCount);
        }
        if !self
            .images
            .iter()
            .zip(images)
            .all(|(cached, incoming)| cached.as_slice() == incoming.bytes.as_slice())
        {
            return Some(HeadMismatch::ImageBytes);
        }
        if self.chunks != chunks {
            return Some(HeadMismatch::ChunkShape);
        }
        if self.n_pos != n_pos {
            return Some(HeadMismatch::Positions);
        }
        None
    }
}

/// A llama context retained across generations, paired with a description of
/// what it holds in the KV cache of sequence 0.
///
/// Invariant: positions `[0, media_pos())` hold `head`, and position
/// `media_pos() + i` holds `tokens[i]` for every `i < tokens.len()`. With no
/// head — the text-only case — that reduces to `tokens` being a plain prefix of
/// the cache. The cache may hold further positions beyond the recorded ones;
/// every reuse path removes everything from its resume position onwards, so
/// unrecorded trailing positions can never be attended to.
pub(super) struct SessionKv {
    /// Declared before `_model`: fields drop in declaration order, so the
    /// context is destroyed before the model allocation it points into.
    ctx: LlamaContext<'static>,
    head: Option<MediaHead>,
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
            head: None,
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

    /// First position `tokens` describes.
    fn media_pos(&self) -> usize {
        self.head.as_ref().map_or(0, |head| head.n_pos)
    }

    /// Every position the cache is known to hold, reusable or not.
    fn occupied(&self) -> usize {
        self.media_pos() + self.tokens.len()
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
    /// The cache already holds everything below this absolute KV position.
    /// Decode resumes there.
    ReusePrefix(usize),
    /// The prompt shares nothing useful with a much larger retained cache. Run
    /// in a throwaway context and leave the retained cache intact.
    SacrificialContext,
}

/// What a retained KV cache offers an incoming prompt.
#[derive(Debug, Clone, Copy)]
pub(super) struct RetainedPrefix<'a> {
    /// Positions `[0, media_pos)` hold media the prompt has *already been
    /// shown* to reproduce. Zero for a text-only cache, and zero whenever the
    /// prompt cannot reproduce the head — in which case `tokens` is empty too,
    /// so nothing behind a head is ever resumed at the wrong position.
    media_pos: usize,
    /// Tokens held at positions `[media_pos, media_pos + tokens.len())`.
    tokens: &'a [LlamaToken],
    /// Every position the cache occupies, reusable or not. Only its size
    /// matters: it decides whether the cache is worth protecting.
    occupied: usize,
    n_ctx: u32,
}

impl<'a> RetainedPrefix<'a> {
    /// A cache the incoming prompt can resume from: either it holds no media,
    /// or the caller has already verified the prompt reproduces the head.
    fn resumable(kv: &'a SessionKv) -> Self {
        Self {
            media_pos: kv.media_pos(),
            tokens: &kv.tokens,
            occupied: kv.occupied(),
            n_ctx: kv.n_ctx,
        }
    }

    /// A cache the incoming prompt cannot resume from. Its size still decides
    /// whether it deserves protection from a small unrelated prompt.
    ///
    /// `occupied` counts KV *positions*, while `is_sacrificial_prompt` weighs it
    /// against a prompt measured in tokens. The two agree for every model this
    /// engine runs today (Gemma included: one position per image token). Under
    /// M-RoPE, where an image occupies fewer positions than it costs tokens, the
    /// comparison undercounts the cache and the protection is weaker than
    /// intended — a small prompt evicts a cache it should have run beside. That
    /// direction is a lost optimisation, never a wrong answer.
    fn opaque(kv: &SessionKv) -> Self {
        Self {
            media_pos: 0,
            tokens: &[],
            occupied: kv.occupied(),
            n_ctx: kv.n_ctx,
        }
    }
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

/// Decide how to prefill a prompt whose plain-token part is `prompt`.
///
/// For a text-only request `prompt` is the whole prompt. For a request that
/// reproduces a retained media head it is only the tail that follows the head,
/// and the returned position is absolute — the head's positions count towards
/// the reuse threshold because they are the expensive ones.
pub(super) fn prefill_plan(
    retained: Option<RetainedPrefix<'_>>,
    prompt: &[LlamaToken],
    effective_ctx: u32,
) -> PrefillPlan {
    let Some(cached) = retained else {
        return PrefillPlan::CreateContext;
    };
    if cached.n_ctx != effective_ctx {
        return PrefillPlan::CreateContext;
    }
    // llama.cpp needs a non-empty batch to produce logits, so at least the
    // final prompt token must always be decoded.
    let reusable = common_prefix_len(cached.tokens, prompt).min(prompt.len().saturating_sub(1));
    let resume = cached.media_pos + reusable;
    if resume >= REUSE_MIN_TOKENS && reusable < prompt.len() {
        return PrefillPlan::ReusePrefix(resume);
    }
    if is_sacrificial_prompt(prompt.len(), cached.occupied) {
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

/// LOAD-BEARING FOR KV REUSE: every partial-resume path in this file depends on
/// `swa_full` being true, so it is pinned here rather than inherited.
///
/// A sliding-window model (Gemma among them) normally keeps only a window of KV
/// per SWA layer, so positions behind `seq_pos_max - window` have already been
/// evicted. Resuming a decode at such a position would attend to KV that is no
/// longer there and quietly produce different output — no error, just a worse
/// answer. `swa_full = true` makes llama.cpp allocate the full-size SWA cache
/// instead, which is what makes `reuse_prefix` (and the multimodal head resume
/// built on it) sound. The text-path prefix reuse has the same dependency.
///
/// The pin costs nothing today: it is also the vendored default
/// (`llama_context_default_params()` in `llama.cpp/src/llama-context.cpp`, and
/// llama-cpp-2 0.1.146's own `swa_full()` doctest asserts it). Stating it in
/// code is what stops a future goose/llama.cpp sync from flipping that default
/// and silently un-sounding every resume path. IF THIS LINE IS EVER REMOVED,
/// partial resume must go with it: `prefill_plan` must stop returning
/// `ReusePrefix` and `reuse_media_head` must stop resuming behind a head.
pub(super) fn build_context_params(
    ctx_size: u32,
    settings: &crate::local_model_registry::ModelSettings,
) -> LlamaContextParams {
    let mut params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(ctx_size))
        .with_swa_full(true);

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

/// How many of `prompt`'s tokens survive a resume at absolute `resume_pos`, or
/// `None` when the position is not one this cache can honour.
///
/// Three bounds, all of them load-bearing:
///
/// - `resume_pos >= media_pos`: a resume inside the media head would land in KV
///   that is not token-addressable at all.
/// - `kept <= prompt_len`: a range bound on `prompt[kept..]`, and nothing more.
///   It does NOT establish that the kept tokens are ones the incoming prompt
///   reproduces — no prefix equality is checked here. That property comes from
///   the caller: `prefill_plan` derives `resume_pos` from a `common_prefix_len`
///   of the cached and incoming tokens. A future caller that computes
///   `resume_pos` some other way must establish prefix equality itself, or it
///   will resume over KV the prompt no longer describes.
/// - `kept <= cached_len`: the caller truncates the token ledger to `kept`, and
///   `Vec::truncate` past the end is a silent no-op. Without this bound a
///   `resume_pos` past the recorded tail would leave `tokens` claiming
///   positions the KV cache was just trimmed of — the ledger and the cache
///   would disagree, which is the precise failure this cache exists to avoid.
///
/// `prefill_plan` never produces a position that violates the last two, since
/// its `reusable` is a common-prefix length of both slices. They are checked
/// anyway because this function is what makes the truncate safe, and the only
/// caller that computes `resume_pos` from something other than a fresh
/// `prefill_plan` would be a future one.
fn kept_prefix_tokens(
    resume_pos: usize,
    media_pos: usize,
    prompt_len: usize,
    cached_len: usize,
) -> Option<usize> {
    resume_pos
        .checked_sub(media_pos)
        .filter(|kept| *kept <= prompt_len && *kept <= cached_len)
}

/// Trim the KV cache back to absolute position `resume_pos` and decode the rest
/// of `prompt` there. `prompt` starts at the cache's `media_pos`, so the tokens
/// it keeps are those before `resume_pos - media_pos`.
///
/// Sound only while contexts are built with `swa_full = true`; see
/// [`build_context_params`].
fn reuse_prefix(
    kv: &mut SessionKv,
    prompt: &[LlamaToken],
    resume_pos: usize,
) -> Result<(), PrefixReuseError> {
    let kept = kept_prefix_tokens(resume_pos, kv.media_pos(), prompt.len(), kv.tokens.len())
        .ok_or(PrefixReuseError::Refused)?;
    let p0 = u32::try_from(resume_pos).map_err(|_| PrefixReuseError::Refused)?;
    match kv.ctx.clear_kv_cache_seq(Some(0), Some(p0), None) {
        Ok(true) => {}
        Ok(false) | Err(_) => return Err(PrefixReuseError::Refused),
    }

    kv.tokens.truncate(kept);
    decode_tokens(&mut kv.ctx, &prompt[kept..], resume_pos).map_err(PrefixReuseError::Poisoned)?;
    kv.tokens.extend_from_slice(&prompt[kept..]);
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
    kv.head = None;
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
        session.as_ref().map(|kv| {
            // A text-only prompt carries no media, so it can never reproduce a
            // retained media head and nothing behind that head is resumable.
            if kv.head.is_some() {
                RetainedPrefix::opaque(kv)
            } else {
                RetainedPrefix::resumable(kv)
            }
        }),
        prompt,
        n_ctx,
    );
    tracing::debug!(
        ?plan,
        cached_tokens = session.as_ref().map(|kv| kv.tokens.len()),
        cached_media_pos = session.as_ref().map(SessionKv::media_pos),
        cached_n_ctx = session.as_ref().map(|kv| kv.n_ctx),
        prompt_tokens = prompt.len(),
        n_ctx,
        "prompt prefill plan"
    );

    match plan {
        PrefillPlan::ReusePrefix(resume) => {
            let kv = session
                .as_mut()
                .expect("ReusePrefix is only produced when a session is retained");
            match reuse_prefix(kv, prompt, resume) {
                Ok(()) => {
                    return Ok(PrefilledPrompt {
                        reused_prefix_tokens: resume,
                        transient: None,
                        effective_ctx,
                    })
                }
                Err(PrefixReuseError::Refused) => {
                    tracing::debug!(
                        resume,
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

/// A prompt's mtmd chunk list, split at its last media chunk.
struct SplitChunks {
    head: Vec<HeadChunk>,
    /// Positions the head occupies. The tail starts here.
    head_pos: usize,
    /// Tokens of every chunk after the last media chunk, concatenated. They
    /// take one position each from `head_pos`, exactly where `mtmd_helper`
    /// would place them.
    tail: Vec<LlamaToken>,
}

/// Describe one chunk, or refuse it.
///
/// `None` means "this chunk cannot be modelled position for position", which
/// makes the whole cache ineligible for reuse. The refusal that fires on real
/// input is the audio chunk, whose position model this code has never
/// exercised. The text-chunk check below is a tripwire, not a live filter:
/// mtmd returns the same value from `get_n_tokens` and `get_n_pos` for a text
/// chunk, so `n_pos != n_tokens` is unreachable today and only earns its place
/// if that ever stops being true.
///
/// # On an unrecognised chunk type
///
/// The `match` below is exhaustive on purpose — no `_` arm. If a future
/// llama-cpp-2 adds a fourth `MtmdInputChunkType`, this stops compiling, which
/// is the outcome we want: someone decides what the new type means for the
/// position model rather than a wildcard silently refusing (or worse,
/// mis-describing) it.
///
/// The other direction — llama.cpp starting to emit a chunk type the *binding*
/// does not know — cannot be turned into a refusal here, and it is worth being
/// precise about why rather than leaving it looking like an oversight:
///
/// - `MtmdInputChunkType::from` panics on an unrecognised discriminant
///   (llama-cpp-2 0.1.146 `mtmd.rs:48`), and the release profile sets
///   `panic = "abort"`, so `catch_unwind` cannot convert it into a refusal.
/// - Reading the discriminant ourselves is not possible either: both
///   `MtmdInputChunks::chunks` and `MtmdInputChunk::chunk` are `pub(crate)`, so
///   there is no way to reach `mtmd_input_chunk_get_type` from outside the
///   crate.
/// - It would not help if there were. The vendored llama.cpp aborts on the same
///   input in three places this code cannot avoid:
///   `mtmd_input_chunk_get_n_tokens` and `mtmd_input_chunk_get_n_pos`
///   (`mtmd.cpp:1187`, `:1199`) and `mtmd_helper_eval_chunk_single`
///   (`mtmd-helper.cpp:378`). The first of those is reached from
///   `MtmdInputChunks::total_tokens()`, which `prefill_multimodal` calls to size
///   the context — before this function runs, and on the pre-cache code path
///   too. The fallback this function's refusal leads to is `eval_chunks`, which
///   aborts as well.
///
/// So an unknown chunk type kills the process either way, at the same point in
/// the same request, with or without this cache. That is a llama.cpp property,
/// not something this file introduced or can contain; the mitigation that does
/// exist is the compile error above, plus keeping the vendored version pinned.
fn describe_chunk(chunk: &MtmdInputChunk) -> Option<HeadChunk> {
    let n_tokens = chunk.n_tokens();
    let n_pos = usize::try_from(chunk.n_positions()).ok()?;
    match chunk.chunk_type() {
        MtmdInputChunkType::Text => {
            let tokens = chunk.text_tokens().unwrap_or(&[]);
            if tokens.len() != n_tokens || n_pos != n_tokens {
                return None;
            }
            Some(HeadChunk::Text(tokens.to_vec()))
        }
        MtmdInputChunkType::Image => Some(HeadChunk::Media { n_tokens, n_pos }),
        MtmdInputChunkType::Audio => None,
    }
}

/// Describe every chunk mtmd produced, or refuse the list.
fn read_chunks(chunks: &MtmdInputChunks) -> Option<Vec<HeadChunk>> {
    let mut described = Vec::with_capacity(chunks.len());
    for index in 0..chunks.len() {
        described.push(describe_chunk(&chunks.get(index)?)?);
    }
    Some(described)
}

/// Split a described chunk list at its last media chunk.
///
/// Everything after that point is plain text, which is the whole reason the
/// reuse path can run without calling mtmd again: llama-cpp-2 0.1.146 exposes
/// no way to evaluate a subset of chunks, but ordinary token decode at an
/// explicit position is exactly what `mtmd_helper` does for text anyway.
fn split_at_last_media(mut chunks: Vec<HeadChunk>) -> Option<SplitChunks> {
    let last_media = chunks
        .iter()
        .rposition(|chunk| matches!(chunk, HeadChunk::Media { .. }))?;
    let rest = chunks.split_off(last_media + 1);
    let head_pos = chunks.iter().map(HeadChunk::n_pos).sum();

    let mut tail = Vec::new();
    for chunk in &rest {
        match chunk {
            HeadChunk::Text(tokens) => tail.extend_from_slice(tokens),
            // Unreachable past the last media chunk; refuse rather than drop it
            // silently, because a dropped chunk would mis-position the tail.
            HeadChunk::Media { .. } => return None,
        }
    }

    Some(SplitChunks {
        head: chunks,
        head_pos,
        tail,
    })
}

/// Try to serve a media-bearing prompt from the retained cache.
///
/// Returns the absolute position decode resumed at. Every refusal leaves the
/// cache exactly as it was found (or drops it, when a failed decode makes its
/// contents untrustworthy) and the caller rebuilds, so a miss only costs the
/// comparison.
///
/// A refusal is logged with the reason it refused. That is the only way the
/// tool-schema flip described on [`HeadMismatch::ChunkShape`] is observable: it
/// silently and permanently ends reuse for a session, and without a reason in
/// the log the symptom is "the cache just stopped working" with nothing to point
/// at.
fn reuse_media_head(
    session: &mut Option<SessionKv>,
    split: &SplitChunks,
    images: &[ExtractedImage],
    n_ctx: u32,
) -> Option<usize> {
    let kv = session.as_mut()?;
    let head = kv.head.as_ref()?;
    if let Some(reason) = head.mismatch(split.head_pos, &split.head, images) {
        tracing::debug!(
            ?reason,
            cached_head_pos = head.n_pos,
            cached_images = head.images.len(),
            prompt_head_pos = split.head_pos,
            prompt_images = images.len(),
            "retained media head does not match this prompt; rebuilding it"
        );
        return None;
    }

    let PrefillPlan::ReusePrefix(resume) =
        prefill_plan(Some(RetainedPrefix::resumable(kv)), &split.tail, n_ctx)
    else {
        return None;
    };

    match reuse_prefix(kv, &split.tail, resume) {
        Ok(()) => Some(resume),
        Err(PrefixReuseError::Refused) => {
            tracing::debug!(
                resume,
                "llama.cpp declined partial KV removal; rebuilding the multimodal prompt"
            );
            None
        }
        Err(PrefixReuseError::Poisoned(e)) => {
            tracing::warn!(error = %e, "suffix decode failed; rebuilding the generation context");
            *session = None;
            None
        }
    }
}

/// Tokenize text + images via mtmd and prefill them.
///
/// When the retained cache already holds this exact image list at these exact
/// positions, only the text that follows the last image is decoded — the encode
/// and the prefill of everything before it are skipped. Otherwise the cache is
/// rebuilt from scratch, which is what this always used to do.
///
/// "This exact image list" is a strict condition and it is the reason the win
/// is narrower than it first looks: a conversation that accumulates pictures
/// misses every turn. See [`MediaHead`] for which conversation shapes actually
/// benefit and which do not.
#[allow(clippy::too_many_arguments)]
fn prefill_multimodal(
    session: &mut Option<SessionKv>,
    model: &Arc<LlamaModel>,
    mtmd_ctx: Option<&MtmdContext>,
    backend: &LlamaCppBackend,
    prompt_text: &str,
    images: &[ExtractedImage],
    context_limit: usize,
    settings: &ModelSettings,
) -> Result<(usize, PrefilledPrompt), ProviderError> {
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

    // Walks every chunk and sums `mtmd_input_chunk_get_n_tokens`, which aborts
    // inside llama.cpp on a chunk type it does not know. It runs before this
    // function describes anything, and it ran on the pre-cache code path too —
    // which is what makes the exhaustive match in `describe_chunk` the only
    // unknown-chunk-type mitigation available here rather than a missing one.
    let prompt_token_count = chunks.total_tokens();

    let n_ctx_train = model.n_ctx_train() as usize;
    let mmproj_overhead = settings.mmproj_size_bytes;
    // The retained window counts towards what this request can spend: it is
    // either kept (the reuse path allocates nothing) or released before its
    // replacement is allocated.
    let memory_max_ctx = estimate_max_context_for_memory(
        model,
        backend,
        mmproj_overhead,
        retained_kv_tokens(session),
    );
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

    let split = read_chunks(&chunks).and_then(split_at_last_media);
    tracing::debug!(
        described = split.is_some(),
        head_pos = split.as_ref().map(|s| s.head_pos),
        tail_tokens = split.as_ref().map(|s| s.tail.len()),
        cached_media_pos = session.as_ref().map(SessionKv::media_pos),
        prompt_tokens = prompt_token_count,
        "multimodal prefill plan"
    );

    if let Some(split) = split.as_ref() {
        if let Some(resume) = reuse_media_head(session, split, images, n_ctx) {
            return Ok((
                prompt_token_count,
                PrefilledPrompt {
                    reused_prefix_tokens: resume,
                    transient: None,
                    effective_ctx,
                },
            ));
        }
    }

    // Release the retained cache before allocating its replacement so the two
    // KV caches are never resident together.
    *session = None;
    let mut kv = SessionKv::create(model, backend, n_ctx, settings)?;

    let n_batch = kv.ctx.n_batch() as i32;
    let n_past = chunks
        .eval_chunks(mtmd_ctx, &kv.ctx, 0, 0, n_batch, true)
        .map_err(|e| ProviderError::ExecutionError(format!("Multimodal eval failed: {e}")))?;

    // Retain the cache only when the chunk list was describable AND the position
    // model it produced is addressable.
    //
    // The `i32::try_from` is the part that carries weight. Positions are
    // `llama_pos`, an i32, everywhere in llama.cpp; a model whose end position
    // does not fit cannot be trimmed back to or resumed at, and `reuse_prefix`
    // would be handed a `resume_pos` it has to refuse anyway. Better to decline
    // retention now than to hold a cache nothing can address.
    //
    // The `== n_past` half is NOT a ledger check, despite reading like one, and
    // should not be relied on as a safety property. `n_past` is llama.cpp's own
    // running sum of the very `n_pos` values this code already read: text chunks
    // accumulate their batched token counts and media chunks add
    // `mtmd_input_chunk_get_n_pos` (`mtmd-helper.cpp`), so the equality holds by
    // construction whenever `eval_chunks` returns `Ok`. It says nothing about
    // *what* is in the cache. It is kept only as a near-free tripwire against a
    // future `mtmd_helper` that stops walking the list early without reporting
    // an error — an eventuality with no other detector here.
    //
    // Failing either test means the cache must never be reused, so it becomes a
    // throwaway context that serves this generation and is then dropped —
    // including the token bookkeeping, which would otherwise record generated
    // tokens against positions the prompt already occupies.
    let retainable = split.as_ref().is_some_and(|split| {
        i32::try_from(split.head_pos + split.tail.len()).is_ok_and(|total| total == n_past)
    });
    if !retainable {
        tracing::debug!(
            n_past,
            "multimodal cache is not addressable by the recorded position model; not retaining it"
        );
        return Ok((
            prompt_token_count,
            PrefilledPrompt {
                reused_prefix_tokens: 0,
                transient: Some(kv),
                effective_ctx,
            },
        ));
    }

    let split = split.expect("retainable is only true when the chunks were described");
    kv.head = Some(MediaHead {
        chunks: split.head,
        n_pos: split.head_pos,
        images: images.iter().map(|img| img.bytes.clone()).collect(),
    });
    kv.tokens = split.tail;
    *session = Some(kv);

    Ok((
        prompt_token_count,
        PrefilledPrompt {
            reused_prefix_tokens: 0,
            transient: None,
            effective_ctx,
        },
    ))
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
                // Logged only once the compact render has actually succeeded.
                // A failed compact render keeps the full one, and the prefix
                // does not move — claiming otherwise would send whoever reads
                // this log hunting a reuse loss that never happened.
                //
                // When it does succeed it rewrites the whole rendered prompt,
                // system prefix included, invalidating every form of KV reuse
                // for the session it fires in: the text path drops to whatever
                // short prefix survives, and a retained media head stops
                // matching outright (`HeadMismatch::ChunkShape`).
                //
                // Worth logging because the trigger is not a setting anyone
                // turned on — the threshold moves with transcript length and
                // image count, so on a small context it can flip partway
                // through a conversation and never flip back, and the only
                // visible symptom is that reuse quietly stopped.
                match apply_template(compact_tools_json) {
                    Ok(compact) => {
                        tracing::debug!(
                            token_count,
                            estimated_image_tokens,
                            token_budget,
                            "prompt exceeds the tool budget; re-rendered with compact tool \
                             schemas (this changed the prompt prefix and ends KV reuse for \
                             this session)"
                        );
                        compact
                    }
                    Err(compact_err) => {
                        tracing::debug!(
                            error = %compact_err,
                            token_count,
                            token_budget,
                            "prompt exceeds the tool budget but the compact render failed; \
                             keeping the full render"
                        );
                        r
                    }
                }
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
                Ok(r) => {
                    tracing::debug!(
                        "rendered with compact tool schemas after the full render failed \
                         (this changed the prompt prefix and ends KV reuse for this session)"
                    );
                    r
                }
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
        prefill_multimodal(
            ctx.session,
            ctx.model,
            ctx.mtmd_ctx,
            ctx.backend,
            &template_result.prompt,
            ctx.images,
            ctx.context_limit,
            ctx.settings,
        )?
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

    /// A text-only retained cache: no media head, so every cached token sits at
    /// its own index.
    fn retained(tokens: &[LlamaToken], n_ctx: u32) -> RetainedPrefix<'_> {
        RetainedPrefix {
            media_pos: 0,
            tokens,
            occupied: tokens.len(),
            n_ctx,
        }
    }

    /// A cache whose media head the incoming prompt reproduces: `tokens` are the
    /// tail that follows `media_pos` positions of media-bearing prefix.
    fn retained_behind_media(
        media_pos: usize,
        tokens: &[LlamaToken],
        n_ctx: u32,
    ) -> RetainedPrefix<'_> {
        RetainedPrefix {
            media_pos,
            tokens,
            occupied: media_pos + tokens.len(),
            n_ctx,
        }
    }

    fn image(bytes: &[u8]) -> ExtractedImage {
        ExtractedImage {
            bytes: bytes.to_vec(),
        }
    }

    fn media_head(chunks: Vec<HeadChunk>, images: &[&[u8]]) -> MediaHead {
        MediaHead {
            n_pos: chunks.iter().map(HeadChunk::n_pos).sum(),
            chunks,
            images: images.iter().map(|bytes| bytes.to_vec()).collect(),
        }
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
            prefill_plan(Some(retained(&cached, 2048)), &prompt, 4096),
            PrefillPlan::CreateContext
        );
    }

    #[test]
    fn prefill_plan_reuses_long_shared_prefix() {
        let cached = tokens(0..1000);
        let prompt = diverging_prompt(900, 40);
        assert_eq!(
            prefill_plan(Some(retained(&cached, 4096)), &prompt, 4096),
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
            prefill_plan(Some(retained(&cached, 4096)), &prompt, 4096),
            PrefillPlan::FullPrefillInPlace
        );
    }

    #[test]
    fn prefill_plan_reuses_at_exactly_the_threshold() {
        let cached = tokens(0..1000);
        let prompt = diverging_prompt(REUSE_MIN_TOKENS, 40);
        assert_eq!(
            prefill_plan(Some(retained(&cached, 4096)), &prompt, 4096),
            PrefillPlan::ReusePrefix(REUSE_MIN_TOKENS)
        );
    }

    #[test]
    fn prefill_plan_always_leaves_a_token_to_decode() {
        let cached = tokens(0..1000);
        let prompt = tokens(0..1000);
        assert_eq!(
            prefill_plan(Some(retained(&cached, 4096)), &prompt, 4096),
            PrefillPlan::ReusePrefix(999)
        );
    }

    #[test]
    fn prefill_plan_reuses_when_prompt_extends_the_cache() {
        let cached = tokens(0..1000);
        let prompt = tokens(0..1400);
        assert_eq!(
            prefill_plan(Some(retained(&cached, 4096)), &prompt, 4096),
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
            prefill_plan(Some(retained(&cached, 4096)), &prompt, 4096),
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
            prefill_plan(Some(retained(&cached, 4096)), &prompt, 4096),
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
            prefill_plan(Some(retained(&cached, 4096)), &prompt, 4096),
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
            prefill_plan(Some(retained(&cached, 4096)), &prompt, 4096),
            PrefillPlan::ReusePrefix(REUSE_MIN_TOKENS)
        );
    }

    #[test]
    fn sacrifice_needs_a_retained_context_of_the_same_size() {
        let cached = tokens(0..7138);
        let prompt = diverging_prompt(8, 370);
        // A window change forces a rebuild regardless of prompt size.
        assert_eq!(
            prefill_plan(Some(retained(&cached, 2048)), &prompt, 4096),
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
            prefill_plan(Some(retained(&[], 4096)), &diverging_prompt(1000, 8), 4096),
            PrefillPlan::FullPrefillInPlace
        );
    }

    #[test]
    fn prefill_plan_on_empty_prompt_prefills_in_place() {
        let cached = tokens(0..1000);
        assert_eq!(
            prefill_plan(Some(retained(&cached, 4096)), &[], 4096),
            PrefillPlan::FullPrefillInPlace
        );
    }

    // --- media head reuse ---

    #[test]
    fn head_chunk_positions_count_text_tokens_and_media_positions() {
        assert_eq!(HeadChunk::Text(tokens(0..7)).n_pos(), 7);
        assert_eq!(
            HeadChunk::Media {
                n_tokens: 256,
                n_pos: 64
            }
            .n_pos(),
            64
        );
    }

    #[test]
    fn split_at_last_media_puts_every_trailing_text_chunk_in_the_tail() {
        let split = split_at_last_media(vec![
            HeadChunk::Text(tokens(0..10)),
            HeadChunk::Media {
                n_tokens: 256,
                n_pos: 256,
            },
            HeadChunk::Text(tokens(100..104)),
            HeadChunk::Text(tokens(200..203)),
        ])
        .expect("a list with a media chunk splits");

        assert_eq!(split.head_pos, 266);
        assert_eq!(split.head.len(), 2);
        assert_eq!(split.tail, tokens((100..104).chain(200..203)));
    }

    #[test]
    fn split_at_last_media_splits_after_the_last_of_several_media_chunks() {
        let media = HeadChunk::Media {
            n_tokens: 64,
            n_pos: 64,
        };
        let split = split_at_last_media(vec![
            HeadChunk::Text(tokens(0..5)),
            media.clone(),
            HeadChunk::Text(tokens(10..15)),
            media,
            HeadChunk::Text(tokens(20..25)),
        ])
        .expect("a list with media chunks splits");

        // Everything up to and including the second image stays in the head, so
        // no image ever has to be re-evaluated on the reuse path.
        assert_eq!(split.head.len(), 4);
        assert_eq!(split.head_pos, 5 + 64 + 5 + 64);
        assert_eq!(split.tail, tokens(20..25));
    }

    #[test]
    fn split_at_last_media_refuses_a_chunk_list_without_media() {
        assert!(split_at_last_media(vec![HeadChunk::Text(tokens(0..10))]).is_none());
        assert!(split_at_last_media(Vec::new()).is_none());
    }

    /// What `reuse_media_head` passes as the incoming head's position count:
    /// `SplitChunks::head_pos`, i.e. the sum over the incoming chunks.
    fn incoming_head_pos(chunks: &[HeadChunk]) -> usize {
        chunks.iter().map(HeadChunk::n_pos).sum()
    }

    fn mismatch(
        head: &MediaHead,
        chunks: &[HeadChunk],
        images: &[ExtractedImage],
    ) -> Option<HeadMismatch> {
        head.mismatch(incoming_head_pos(chunks), chunks, images)
    }

    #[test]
    fn media_head_matches_identical_shape_and_images() {
        let head = media_head(
            vec![
                HeadChunk::Text(tokens(0..10)),
                HeadChunk::Media {
                    n_tokens: 256,
                    n_pos: 256,
                },
            ],
            &[b"png-bytes"],
        );
        let same = head.chunks.clone();
        assert_eq!(mismatch(&head, &same, &[image(b"png-bytes")]), None);
    }

    #[test]
    fn media_head_rejects_different_image_bytes() {
        let head = media_head(
            vec![HeadChunk::Media {
                n_tokens: 256,
                n_pos: 256,
            }],
            &[b"first-picture"],
        );
        let same_shape = head.chunks.clone();
        // Same chunk shape, different picture: the token counts cannot tell
        // these apart, so the bytes have to.
        assert_eq!(
            mismatch(&head, &same_shape, &[image(b"second-picture")]),
            Some(HeadMismatch::ImageBytes)
        );
    }

    /// One image's worth of chunks, at the size mtmd gives a fixed resolution.
    fn media_chunk() -> HeadChunk {
        HeadChunk::Media {
            n_tokens: 256,
            n_pos: 256,
        }
    }

    /// The head a single-image prompt produces: a preamble, then the picture.
    fn one_image_head() -> MediaHead {
        media_head(vec![HeadChunk::Text(tokens(0..10)), media_chunk()], &[b"a"])
    }

    #[test]
    fn media_head_rejects_a_grown_image_list() {
        // The growing-image-list conversation, shaped the way mtmd actually
        // presents it: a second picture arrives as a second media chunk *and* a
        // second image, so the chunk list and the position count move with the
        // count. All three checks would fire; ImageCount is the narrowest, and
        // it is the reason the trace that motivated this cache gets nothing
        // from it.
        let head = one_image_head();
        let two_images = vec![
            HeadChunk::Text(tokens(0..10)),
            media_chunk(),
            HeadChunk::Text(tokens(20..24)),
            media_chunk(),
        ];
        assert_eq!(
            mismatch(&head, &two_images, &[image(b"a"), image(b"b")]),
            Some(HeadMismatch::ImageCount)
        );
    }

    #[test]
    fn media_head_rejects_a_rewritten_preamble() {
        // The tool-schema flip: same picture, rewritten text ahead of it. The
        // compact schema renders shorter, so the image moves too — but the
        // chunk list is the narrower way to say that, and it is what gets
        // logged.
        let head = one_image_head();
        let shorter = vec![HeadChunk::Text(tokens(500..506)), media_chunk()];
        assert_eq!(
            mismatch(&head, &shorter, &[image(b"a")]),
            Some(HeadMismatch::ChunkShape)
        );

        // A rewrite that happens to render to the same token count is caught by
        // the same check, and only because chunks are compared token for token.
        // Catching it is what keeps the resume from attending to a preamble the
        // prompt no longer contains.
        let same_length = vec![HeadChunk::Text(tokens(500..510)), media_chunk()];
        assert_eq!(incoming_head_pos(&same_length), head.n_pos);
        assert_eq!(
            mismatch(&head, &same_length, &[image(b"a")]),
            Some(HeadMismatch::ChunkShape)
        );
    }

    #[test]
    fn media_head_rejects_a_recorded_size_that_disagrees_with_its_chunks() {
        // Deliberately not an input mtmd can produce: `head_pos` is the sum over
        // exactly these chunks on both sides, so an identical chunk list always
        // means an identical position count. This is the tripwire for a
        // `MediaHead` built with an `n_pos` that does not describe its own
        // chunks — the one state in which a matching head would resume the
        // decode at the wrong absolute position.
        let mut head = one_image_head();
        head.n_pos += 1;
        let same = head.chunks.clone();
        assert_eq!(
            mismatch(&head, &same, &[image(b"a")]),
            Some(HeadMismatch::Positions)
        );
    }

    // --- the bound that keeps the token ledger and the KV cache in step ---

    #[test]
    fn kept_prefix_tokens_counts_from_the_end_of_the_media_head() {
        assert_eq!(kept_prefix_tokens(1040, 1000, 52, 90), Some(40));
        // Text-only cache: the resume position is the kept count.
        assert_eq!(kept_prefix_tokens(900, 0, 1000, 1000), Some(900));
    }

    #[test]
    fn kept_prefix_tokens_refuses_a_resume_inside_the_media_head() {
        assert_eq!(kept_prefix_tokens(999, 1000, 90, 90), None);
    }

    #[test]
    fn kept_prefix_tokens_refuses_a_resume_past_the_incoming_prompt() {
        assert_eq!(kept_prefix_tokens(1050, 1000, 40, 90), None);
    }

    #[test]
    fn kept_prefix_tokens_refuses_a_resume_past_the_recorded_tail() {
        // The bound that matters most: `Vec::truncate` past the end is a silent
        // no-op, so allowing this would leave the ledger claiming tokens at
        // positions `clear_kv_cache_seq` had just removed.
        assert_eq!(kept_prefix_tokens(1090, 1000, 200, 80), None);
        // Exactly at the end of the tail is fine — nothing is dropped.
        assert_eq!(kept_prefix_tokens(1080, 1000, 200, 80), Some(80));
    }

    #[test]
    fn prefill_plan_resumes_at_an_absolute_position_behind_a_media_head() {
        // Turn 2 of an image conversation: the head and the first 40 tail tokens
        // are already in the cache, so decode resumes at 1000 + 40.
        let cached = tokens(0..90);
        let prompt = diverging_prompt(40, 12);
        assert_eq!(
            prefill_plan(
                Some(retained_behind_media(1000, &cached, 4096)),
                &prompt,
                4096
            ),
            PrefillPlan::ReusePrefix(1040)
        );
    }

    #[test]
    fn a_media_head_alone_can_carry_the_prompt_over_the_reuse_threshold() {
        // Only two tail tokens match, but they sit behind 900 positions of image
        // that would otherwise be re-encoded and re-prefilled from scratch.
        let cached = tokens(0..80);
        let prompt = diverging_prompt(2, 30);
        assert_eq!(
            prefill_plan(
                Some(retained_behind_media(900, &cached, 4096)),
                &prompt,
                4096
            ),
            PrefillPlan::ReusePrefix(902)
        );
    }

    #[test]
    fn a_media_head_too_short_to_reach_the_threshold_does_not_reuse() {
        // The head is real but small, and the tail shares nothing with it, so
        // the resume position stays under REUSE_MIN. Sized so the cache is not
        // worth sacrificing for either, which isolates the threshold.
        let cached = tokens(0..80);
        let prompt = diverging_prompt(0, 180);
        assert_eq!(
            prefill_plan(
                Some(retained_behind_media(100, &cached, 4096)),
                &prompt,
                4096
            ),
            PrefillPlan::FullPrefillInPlace
        );
    }

    #[test]
    fn a_media_cache_the_prompt_cannot_reproduce_offers_no_tokens() {
        // What `RetainedPrefix::opaque` produces: a text-only prompt facing a
        // media cache. The tail tokens must not be matched against it, because
        // they sit at positions the text prompt would not put them at.
        let opaque = RetainedPrefix {
            media_pos: 0,
            tokens: &[],
            occupied: 3000,
            n_ctx: 4096,
        };
        let prompt = tokens(0..2000);
        assert_eq!(
            prefill_plan(Some(opaque), &prompt, 4096),
            PrefillPlan::FullPrefillInPlace
        );
    }

    #[test]
    fn a_side_call_is_sacrificed_against_the_positions_a_media_cache_occupies() {
        // The memory-extraction interleave, now against an image conversation.
        // `occupied` counts the media head, so the expensive cache is protected
        // even though its recorded token tail is short.
        let opaque = RetainedPrefix {
            media_pos: 0,
            tokens: &[],
            occupied: 3000,
            n_ctx: 4096,
        };
        let prompt = tokens(0..378);
        assert_eq!(
            prefill_plan(Some(opaque), &prompt, 4096),
            PrefillPlan::SacrificialContext
        );
    }

    #[test]
    fn a_media_cache_of_a_different_window_is_rebuilt() {
        let cached = tokens(0..90);
        let prompt = diverging_prompt(40, 12);
        assert_eq!(
            prefill_plan(
                Some(retained_behind_media(1000, &cached, 2048)),
                &prompt,
                4096
            ),
            PrefillPlan::CreateContext
        );
    }

    #[test]
    fn reuse_behind_a_media_head_always_leaves_a_token_to_decode() {
        // The tail is fully cached. llama.cpp still needs a non-empty batch for
        // logits, so the last tail token is re-decoded.
        let cached = tokens(0..90);
        let prompt = tokens(0..90);
        assert_eq!(
            prefill_plan(
                Some(retained_behind_media(1000, &cached, 4096)),
                &prompt,
                4096
            ),
            PrefillPlan::ReusePrefix(1089)
        );
    }

    #[test]
    fn an_empty_tail_behind_a_media_head_is_never_reused() {
        // No token left to decode means no logits, so the prompt has to be
        // rebuilt rather than resumed at the end of the head.
        assert_eq!(
            prefill_plan(Some(retained_behind_media(1000, &[], 4096)), &[], 4096),
            PrefillPlan::FullPrefillInPlace
        );
    }
}
