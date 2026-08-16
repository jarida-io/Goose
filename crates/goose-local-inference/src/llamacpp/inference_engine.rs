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
    /// On-disk preamble cache for this model. `None` disables snapshots for the
    /// request without changing any other behaviour.
    pub snapshot: &'a mut Option<super::prompt_snapshot::SnapshotSlot>,
    /// The GGUF this slot loaded, for keying that cache.
    pub model_path: &'a std::path::Path,
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
    /// On-disk preamble cache. Built lazily on the first prefill, because its
    /// key includes `n_ctx` and that is a property of the request rather than
    /// of the load.
    pub snapshot: Option<super::prompt_snapshot::SnapshotSlot>,
    /// Where this model was loaded from. Part of the snapshot key.
    pub model_path: std::path::PathBuf,
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

/// Trim a retained cache to its first `keep` token positions.
///
/// Only valid for a text-only cache: with a media head the positions `tokens`
/// describes start at `media_pos`, and a caller passing a token count as an
/// absolute position would cut the head in half.
fn trim_to_prefix(kv: &mut SessionKv, keep: usize) -> Result<(), ()> {
    debug_assert!(kv.head.is_none(), "trim_to_prefix is text-only");
    if keep > kv.tokens.len() {
        return Err(());
    }
    let p0 = u32::try_from(keep).map_err(|_| ())?;
    match kv.ctx.clear_kv_cache_seq(Some(0), Some(p0), None) {
        Ok(true) => {
            kv.tokens.truncate(keep);
            Ok(())
        }
        Ok(false) | Err(_) => Err(()),
    }
}

/// Build a fresh context and try to fill its preamble from disk.
///
/// `Ok(Some(n))` means the context now holds the whole prompt with `n` leading
/// positions restored rather than decoded. `Ok(None)` means the context is
/// fresh and empty and the caller should prefill it as usual -- the context is
/// left in `session` either way, so no allocation is wasted.
fn seed_from_snapshot(
    session: &mut Option<SessionKv>,
    model: &Arc<LlamaModel>,
    backend: &LlamaCppBackend,
    settings: &ModelSettings,
    slot: &mut super::prompt_snapshot::SnapshotSlot,
    prompt: &[LlamaToken],
    n_ctx: u32,
) -> Result<Option<usize>, ProviderError> {
    // Released before the replacement is allocated, so two KV caches are never
    // resident at once -- the same rule `full_prefill` follows, and on an 8 GB
    // device the difference between fitting and an allocation failure.
    *session = None;
    let mut kv = SessionKv::create(model, backend, n_ctx, settings)?;

    let Some(restored) = slot.load(kv.context_mut(), prompt) else {
        *session = Some(kv);
        return Ok(None);
    };

    kv.tokens.extend_from_slice(&prompt[..restored]);
    match decode_tokens(&mut kv.ctx, &prompt[restored..], restored) {
        Ok(()) => {
            kv.tokens.extend_from_slice(&prompt[restored..]);
            *session = Some(kv);
            // Distinct from "loaded": that says bytes came off disk, this says a
            // turn is actually generating on them. A restore that loads and then
            // fails its suffix decode is a load without a use, and the two need
            // telling apart when reading back why a turn was slow.
            tracing::info!(
                target: "giap::kv",
                event = "in_use",
                restored_tokens = restored,
                decoded_tokens = prompt.len() - restored,
                prompt_tokens = prompt.len(),
                "kv snapshot in use; only the suffix was decoded"
            );
            Ok(Some(restored))
        }
        Err(e) => {
            // A restored cache that will not decode is worse than none: drop it
            // and let the caller build a clean one.
            tracing::warn!(
                target: "giap::kv",
                event = "discarded",
                reason = "suffix decode failed after restore",
                error = %e,
                "kv snapshot discarded; rebuilding the context"
            );
            *session = None;
            Err(e)
        }
    }
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

/// Persist this prompt's shape once its stable prefix is known.
///
/// Called after the cache holds the whole prompt, whichever plan put it there.
/// That placement is the point, and two earlier ones were wrong: hanging it off
/// `ReusePrefix` meant only a shape that already owned the retained cache could
/// write, and requiring the cache to ALREADY hold the prefix failed the moment
/// two callers alternated — which is every real pond, where chat, extraction,
/// titling and the reviewer take turns against one model and share nothing at
/// the start. Both produced an empty cache directory.
///
/// The cache is trimmed to the prefix, saved, and the tail decoded back. That
/// re-decode is the whole cost, it is bounded by the divergent tail rather than
/// the prompt, and it happens once per shape per process — against a saved cold
/// start measured in seconds.
fn write_snapshot_if_due(
    session: &mut Option<SessionKv>,
    snapshot: &mut Option<super::prompt_snapshot::SnapshotSlot>,
    prompt: &[LlamaToken],
) {
    let (Some(slot), Some(kv)) = (snapshot.as_mut(), session.as_mut()) else {
        return;
    };
    // Text-only: with a media head `tokens` starts at `media_pos`, so a token
    // count is not an absolute position.
    if kv.head.is_some() {
        return;
    }
    let Some(keep) = slot.writable_prefix(prompt) else {
        return;
    };
    // The cache has to hold what is about to be written, and there has to be a
    // tail worth keeping the rest of.
    if keep >= kv.tokens.len() || common_prefix_len(&kv.tokens, prompt) < keep {
        return;
    }
    if trim_to_prefix(kv, keep).is_err() {
        return;
    }
    slot.write(&kv.ctx, &kv.tokens);

    // Put the tail back: the turn still has to generate from the whole prompt.
    match decode_tokens(&mut kv.ctx, &prompt[keep..], keep) {
        Ok(()) => kv.tokens.extend_from_slice(&prompt[keep..]),
        Err(e) => {
            tracing::warn!(
                target: "giap::kv",
                event = "discarded",
                reason = "re-decoding the tail after a write failed",
                error = %e,
                "kv snapshot written but the context could not be restored; rebuilding"
            );
            *session = None;
        }
    }
}

/// Prepare a KV cache holding `prompt`, with logits available for its last
/// token.
#[allow(clippy::too_many_arguments)]
fn prefill_prompt(
    session: &mut Option<SessionKv>,
    snapshot: &mut Option<super::prompt_snapshot::SnapshotSlot>,
    model_path: &std::path::Path,
    model: &Arc<LlamaModel>,
    backend: &LlamaCppBackend,
    settings: &ModelSettings,
    prompt: &[LlamaToken],
    effective_ctx: usize,
) -> Result<PrefilledPrompt, ProviderError> {
    let n_ctx = u32::try_from(effective_ctx).map_err(|_| {
        ProviderError::ExecutionError(format!("Context size {effective_ctx} exceeds u32 range"))
    })?;

    // The key includes `n_ctx`, which is a property of the request rather than of
    // the load, so the slot is built here and rebuilt whenever the window moves.
    // A stale-window slot would name a file whose cache has the wrong shape.
    if snapshot
        .as_ref()
        .is_none_or(|slot| !slot.matches(model_path, n_ctx))
    {
        *snapshot = super::prompt_snapshot::SnapshotSlot::for_model(model_path, n_ctx, settings);
    }
    if let Some(slot) = snapshot.as_mut() {
        slot.observe(prompt);
    }

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

    // Nothing usable in memory: this is the cold start the snapshot exists for.
    if matches!(plan, PrefillPlan::CreateContext) {
        if let Some(slot) = snapshot.as_mut() {
            match seed_from_snapshot(session, model, backend, settings, slot, prompt, n_ctx) {
                Ok(Some(restored)) => {
                    return Ok(PrefilledPrompt {
                        reused_prefix_tokens: restored,
                        transient: None,
                        effective_ctx,
                    })
                }
                // No usable file. `session` now holds a fresh empty context that
                // `full_prefill` below will decode into rather than reallocate.
                Ok(None) => {}
                Err(e) => {
                    tracing::debug!(error = %e, "snapshot restore failed; prefilling normally");
                }
            }
        }
    }

    match plan {
        PrefillPlan::ReusePrefix(resume) => {
            let kv = session
                .as_mut()
                .expect("ReusePrefix is only produced when a session is retained");

            match reuse_prefix(kv, prompt, resume) {
                Ok(()) => {
                    write_snapshot_if_due(session, snapshot, prompt);
                    return Ok(PrefilledPrompt {
                        reused_prefix_tokens: resume,
                        transient: None,
                        effective_ctx,
                    });
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
    write_snapshot_if_due(session, snapshot, prompt);
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
            ctx.snapshot,
            ctx.model_path,
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

    /// The snapshot is only worth anything if a restored cache answers
    /// IDENTICALLY to one that was decoded from scratch.
    ///
    /// That is the whole risk of this feature and the one thing no unit test can
    /// reach: a KV blob restored into a mismatched context does not error, it
    /// produces plausible tokens that are subtly wrong. So the assertion is not
    /// "the file loaded" -- it is that the next-token logits after
    /// restore-then-decode-the-tail are bit-identical to full prefill.
    ///
    /// `#[ignore]` because it needs a real GGUF; run with `--ignored`. It skips
    /// rather than fails when the model is absent, so the suite stays green on a
    /// machine that has never downloaded one.
    ///
    /// The model is HARD-LINKED into the temp dir, never symlinked. A symlinked
    /// models directory has twice caused a scratch run to reach through and
    /// destroy a real pond's weights; a hard link is a second directory entry
    /// for the same inode, so nothing this test does can touch the original.
    #[test]
    #[ignore = "requires a real GGUF on disk; run with --ignored"]
    fn a_restored_snapshot_answers_identically_to_a_full_prefill() {
        use crate::llamacpp::prompt_snapshot::SnapshotSlot;
        use llama_cpp_2::model::params::LlamaModelParams;
        use llama_cpp_2::model::AddBos;

        let Some(src) = live_model_path() else {
            eprintln!("skipping: no local GGUF found");
            return;
        };

        let tmp = tempfile::tempdir().expect("tempdir");
        let model_path = tmp.path().join("model.gguf");
        std::fs::hard_link(&src, &model_path).expect("hard link the model, never symlink it");
        // Snapshots land under the temp root instead of the real goose data dir.
        // Process-wide, so it is taken under the same lock the registry tests
        // use rather than set directly — two live tests otherwise race and each
        // reads the other's cache directory.
        let _env = env_lock::lock_env([("GOOSE_PATH_ROOT", Some(tmp.path().to_str().unwrap()))]);

        // GPU offload is opt-in via env so the test is meaningful on both a
        // CPU-only build and a CUDA one. It matters for the timing and not at
        // all for the equivalence claim: the noise-floor control is measured on
        // whatever build is running.
        // UNSET means "whatever llama.cpp would do", which is Metal offload on a
        // Mac. An earlier version defaulted this to 0 and silently turned the
        // GPU off, quadrupling the Mac's prefill and making the comparison a
        // measurement of the harness rather than of the cache.
        let n_gpu_layers: Option<u32> = std::env::var("GIAP_TEST_NGL")
            .ok()
            .and_then(|v| v.parse().ok());
        eprintln!("n_gpu_layers: {n_gpu_layers:?} (None = llama.cpp default)");
        let backend = shared_backend();
        let model = Arc::new(
            LlamaModel::load_from_file(
                backend.llama_backend(),
                &model_path,
                &match n_gpu_layers {
                    Some(n) => LlamaModelParams::default().with_n_gpu_layers(n),
                    None => LlamaModelParams::default(),
                },
            )
            .expect("load model"),
        );

        let settings = ModelSettings {
            n_gpu_layers,
            ..Default::default()
        };
        let n_ctx: u32 = 4096;

        // The preamble has to clear SNAPSHOT_MIN_TOKENS or nothing is written.
        let preamble = "The pond keeps its own notes about the household. ".repeat(120);
        let prefix = model
            .str_to_token(&preamble, AddBos::Always)
            .expect("tokenize preamble");
        assert!(
            prefix.len() > 1024,
            "preamble must exceed the snapshot threshold, got {}",
            prefix.len()
        );
        let tail = model
            .str_to_token(" In one word, the weather today is", AddBos::Never)
            .expect("tokenize tail");
        let prompt: Vec<LlamaToken> = prefix.iter().chain(tail.iter()).copied().collect();

        let mut slot =
            SnapshotSlot::for_model(&model_path, n_ctx, &settings).expect("slot for a real model");

        // 1. Write a snapshot from a cache holding exactly the preamble.
        {
            let mut kv = SessionKv::create(&model, backend, n_ctx, &settings).expect("ctx");
            decode_tokens(&mut kv.ctx, &prefix, 0).expect("prefill preamble");
            kv.tokens.extend_from_slice(&prefix);
            assert!(slot.write(&kv.ctx, &kv.tokens), "snapshot must be written");
        }
        let snapshot_path = slot
            .snapshot_paths()
            .into_iter()
            .next()
            .expect("exactly one snapshot was written");
        let snapshot_bytes = std::fs::metadata(&snapshot_path)
            .expect("snapshot on disk")
            .len();
        eprintln!(
            "snapshot: {} tokens, {:.1} MiB ({:.1} KiB/token)",
            prefix.len(),
            snapshot_bytes as f64 / 1_048_576.0,
            snapshot_bytes as f64 / 1024.0 / prefix.len() as f64
        );

        // 2. Restore into a fresh context and decode only the tail.
        let restored = {
            let started = std::time::Instant::now();
            let mut kv = SessionKv::create(&model, backend, n_ctx, &settings).expect("ctx");
            let n = slot
                .load(kv.context_mut(), &prompt)
                .expect("the snapshot must load and prefix this prompt");
            assert_eq!(n, prefix.len(), "the whole preamble should be restored");
            // The last token is decoded on its own so the logits row is index 0
            // on both paths. `get_logits_ith` validates the index against the
            // final batch, and the two paths batch differently -- restore
            // decodes only the tail, full prefill chunks the whole prompt -- so
            // any fixed non-zero index would be right for one and a panic for
            // the other.
            let last = prompt.len() - 1;
            decode_tokens(&mut kv.ctx, &prompt[n..last], n).expect("decode tail");
            decode_tokens(&mut kv.ctx, &prompt[last..], last).expect("decode final token");
            eprintln!(
                "restore + tail decode: {} ms",
                started.elapsed().as_millis()
            );
            kv.ctx.get_logits_ith(0).to_vec()
        };

        // 3. The control: the same prompt, decoded from nothing.
        let fresh = {
            let started = std::time::Instant::now();
            let mut kv = SessionKv::create(&model, backend, n_ctx, &settings).expect("ctx");
            let last = prompt.len() - 1;
            decode_tokens(&mut kv.ctx, &prompt[..last], 0).expect("full prefill");
            decode_tokens(&mut kv.ctx, &prompt[last..], last).expect("decode final token");
            eprintln!(
                "full prefill:          {} ms",
                started.elapsed().as_millis()
            );
            kv.ctx.get_logits_ith(0).to_vec()
        };

        assert_eq!(
            restored.len(),
            fresh.len(),
            "logit vectors must describe the same vocabulary"
        );
        let argmax = |v: &[f32]| {
            v.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
        };
        assert_eq!(
            argmax(&restored),
            argmax(&fresh),
            "a restored cache predicted a different next token than a full prefill -- the \
             snapshot is not equivalent to the decode it replaces"
        );

        // NOT bitwise, and the first version of this test was wrong to demand it.
        // The two paths batch differently by construction -- restore decodes an
        // 8-token tail, full prefill chunks ~1,200 tokens -- and llama.cpp's
        // reduction order follows the batch shape, so the float results differ
        // in the last places on Metal even when the cache is perfect. Measured
        // here: same argmax, same top five, deltas in the 1e-2 range against
        // logits spanning ~20.
        //
        // What has to hold is that the distribution is the same distribution.
        // The top-5 check is the part that catches a subtly wrong cache: one
        // that agrees on the winner by luck will not agree on the next four.
        let top5 = |v: &[f32]| {
            let mut idx: Vec<usize> = (0..v.len()).collect();
            idx.sort_by(|a, b| {
                v[*b]
                    .partial_cmp(&v[*a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            idx.truncate(5);
            idx
        };
        assert_eq!(
            top5(&restored),
            top5(&fresh),
            "a restored cache ranked the top tokens differently from a full prefill"
        );

        // THE CONTROL, and without it the delta below cannot be interpreted.
        //
        // It decodes the same prompt with the SAME batch shape the restore path
        // uses -- prefix in one call, then the tail -- but computes the prefix
        // instead of restoring it. Whatever this differs from `fresh` by is pure
        // batch-shape float noise for this host, with no snapshot involved.
        //
        // A tolerance guessed on one machine is worthless here: the divergence is
        // 0.007 on Metal and larger on a CPU build, because llama.cpp's
        // accumulation order follows both the batch shape and the SIMD path. So
        // the snapshot is judged against the noise floor measured on the SAME
        // host in the SAME run, not against a constant.
        let control = {
            let mut kv = SessionKv::create(&model, backend, n_ctx, &settings).expect("ctx");
            let last = prompt.len() - 1;
            decode_tokens(&mut kv.ctx, &prefix, 0).expect("prefix");
            decode_tokens(&mut kv.ctx, &prompt[prefix.len()..last], prefix.len()).expect("tail");
            decode_tokens(&mut kv.ctx, &prompt[last..], last).expect("final");
            kv.ctx.get_logits_ith(0).to_vec()
        };

        let spread = |a: &[f32], b: &[f32]| {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max)
        };
        let restored_delta = spread(&restored, &fresh);
        let noise_floor = spread(&control, &fresh);
        eprintln!("max |logit delta| restored vs full prefill: {restored_delta:.6}");
        eprintln!("max |logit delta| CONTROL  vs full prefill: {noise_floor:.6}  (batch-shape noise, no snapshot)");

        assert_eq!(
            top5(&restored),
            top5(&control),
            "restore and an equivalent computed prefix must rank identically"
        );
        // Generous multiplier on purpose: the claim being tested is "the same
        // order of magnitude as re-decoding", not "bitwise", and a snapshot that
        // was actually wrong lands orders of magnitude out, not 3x.
        assert!(
            restored_delta <= (noise_floor * 3.0).max(0.01),
            "restored logits diverge by {restored_delta}, well beyond this host's own \
             batch-shape noise floor of {noise_floor} -- the cache is not equivalent to the \
             decode it replaces"
        );

        // The production failure, pinned. `max_tokens` on `state_seq_load_file`
        // sizes the OUTPUT BUFFER; it is not a filter on what may be read. I read
        // it as a filter and passed the prompt length, so on a live pond a
        // 5,341-token snapshot met a 5,033-token prompt and llama.cpp refused the
        // whole file with "token count in sequence state file exceeded capacity"
        // — on every cold start, forever, because nothing retired the file.
        //
        // A snapshot LONGER than the current prompt is an ordinary thing to find
        // (a later session carries less history), and it belongs in the prefix
        // test where it can be rejected for a stated reason. So the capacity has
        // to be the context's, and this asserts the semantics both ways round.
        {
            let mut kv = SessionKv::create(&model, backend, n_ctx, &settings).expect("ctx");
            let too_small = prefix.len() - 1;
            assert!(
                kv.context_mut()
                    .state_seq_load_file(&snapshot_path, 0, too_small)
                    .is_err(),
                "a capacity below the stored token count must fail -- if this starts passing, \
                 max_tokens has stopped being a buffer size and the reasoning above is stale"
            );

            let mut kv = SessionKv::create(&model, backend, n_ctx, &settings).expect("ctx");
            let (tokens, _) = kv
                .context_mut()
                .state_seq_load_file(&snapshot_path, 0, n_ctx as usize)
                .expect("the context's own capacity must always be enough");
            assert_eq!(
                tokens.len(),
                prefix.len(),
                "the whole stored prefix must come back when the buffer can hold it"
            );
        }

        // The safety half, and the one that decides whether this feature can ship:
        // a snapshot that is NOT a prefix of the incoming prompt must be refused,
        // not restored. This is the shape a changed system prompt or a changed
        // tool set produces -- neither of which the file key can see -- and
        // restoring across it would not be slow, it would be wrong.
        let mut foreign: Vec<LlamaToken> = prompt.clone();
        foreign[0] = LlamaToken(if foreign[0].0 == 1 { 2 } else { 1 });
        let mut kv = SessionKv::create(&model, backend, n_ctx, &settings).expect("ctx");
        assert!(
            slot.load(kv.context_mut(), &foreign).is_none(),
            "a snapshot that does not prefix the prompt must be refused, not restored"
        );
    }

    /// DIAGNOSTIC: does turning thinking off change what the model is told about
    /// tools?
    ///
    /// Observed on a live pond: turns that reason call a tool 81% of the time,
    /// turns that do not call one 27% of the time. That is a 3x gap and the
    /// question is whether the prompt is the cause. `enable_thinking` reaches
    /// llama.cpp twice -- as the flag itself and as `reasoning_format` -- and
    /// both feed the chat template, so the template is where a difference would
    /// live.
    ///
    /// Renders the SAME messages and the SAME tools both ways and compares.
    #[test]
    #[ignore = "requires a real GGUF on disk; run with --ignored"]
    fn thinking_off_does_not_change_what_the_model_is_told_about_tools() {
        use llama_cpp_2::model::params::LlamaModelParams;

        let Some(src) = live_model_path() else {
            eprintln!("skipping: no local GGUF found");
            return;
        };
        let backend = shared_backend();
        let model =
            LlamaModel::load_from_file(backend.llama_backend(), &src, &LlamaModelParams::default())
                .expect("load model");

        let settings = ModelSettings::default();
        let templates = crate::llamacpp::load_chat_templates(&model, &settings).expect("templates");
        let template = templates
            .tool_use
            .as_ref()
            .or(templates.default.as_ref())
            .expect("a template");

        let messages = r#"[{"role":"system","content":"You are a helpful assistant."},
                           {"role":"user","content":"What is the weather in Nairobi?"}]"#;
        let tools = r#"[{"type":"function","function":{"name":"get_current_weather",
                          "description":"Current weather for a place.",
                          "parameters":{"type":"object","properties":{"location":{"type":"string"}},
                          "required":["location"]}}}]"#;

        let render = |thinking: bool| {
            let params = OpenAIChatTemplateParams {
                messages_json: messages,
                tools_json: Some(tools),
                tool_choice: None,
                json_schema: None,
                grammar: None,
                reasoning_format: if thinking { Some("auto") } else { None },
                chat_template_kwargs: None,
                add_generation_prompt: true,
                use_jinja: true,
                parallel_tool_calls: false,
                enable_thinking: thinking,
                add_bos: false,
                add_eos: false,
                parse_tool_calls: true,
            };
            model
                .apply_chat_template_oaicompat(template, &params)
                .expect("apply template")
        };

        let on = render(true);
        let off = render(false);

        // The prompt is only half the question. The template also decides how a
        // RESPONSE is parsed -- chat_format, the tool-call PEG parser, the
        // grammar and its triggers. If those differ, a model emitting a
        // perfectly good tool call with thinking off could have it dropped on
        // the way back, which is a bug rather than a model floor.
        let describe = |label: &str, r: &llama_cpp_2::model::ChatTemplateResult| {
            eprintln!(
                "{label}: chat_format={} grammar={} lazy={} triggers={} parser={} stops={} preserved={} gen_prompt={:?}",
                r.chat_format,
                r.grammar.as_deref().map_or(0, str::len),
                r.grammar_lazy,
                r.grammar_triggers.len(),
                r.parser.as_deref().map_or(0, str::len),
                r.additional_stops.len(),
                r.preserved_tokens.len(),
                r.generation_prompt,
            );
        };
        describe("ON ", &on);
        describe("OFF", &off);
        assert_eq!(
            on.chat_format, off.chat_format,
            "the response PARSER differs between thinking modes; a tool call emitted with \
             thinking off would be parsed by a different format than the one that produced it"
        );

        eprintln!("--- thinking ON  : {} chars", on.prompt.len());
        eprintln!("--- thinking OFF : {} chars", off.prompt.len());
        let needle = "get_current_weather";
        eprintln!(
            "tool name present -> on: {}  off: {}",
            on.prompt.matches(needle).count(),
            off.prompt.matches(needle).count()
        );
        if on.prompt != off.prompt {
            // Show the first divergence with a little context, which is the
            // whole point of the diagnostic.
            // Counted and sliced in CHARS, not bytes: a prompt is model-authored
            // text and a byte range can land inside a UTF-8 sequence.
            let at = on
                .prompt
                .chars()
                .zip(off.prompt.chars())
                .position(|(a, b)| a != b)
                .unwrap_or(0);
            let window =
                |s: &str| -> String { s.chars().skip(at.saturating_sub(160)).take(400).collect() };
            eprintln!("--- first divergence at char {at} ---");
            eprintln!("ON : ...{}", window(&on.prompt));
            eprintln!("OFF: ...{}", window(&off.prompt));
        }

        assert!(
            off.prompt.contains(needle),
            "the tool schema vanished from the prompt when thinking was turned off -- that \
             would explain tool calls only working with thinking on"
        );
    }

    /// A REAL turn must actually write a file. This is the defect Jerry reported:
    /// everything above passed and the cache directory stayed empty.
    ///
    /// Drives `prefill_prompt` — the production entry point — rather than
    /// `slot.write` directly, because the bug was never in writing. It was in
    /// WHEN writing was attempted: the hook hung off the `ReusePrefix` arm, which
    /// only fires when a retained cache already shares the prompt's opening.
    /// Chat reaches it on its second turn; memory extraction takes a sacrificial
    /// context and never does; a quarter-hourly reviewer finds chat's cache in
    /// the slot instead of its own. So one shape could write and the rest never
    /// could, and a test that called `write` itself could not see that.
    #[test]
    #[ignore = "requires a real GGUF on disk; run with --ignored"]
    fn two_turns_of_one_shape_leave_a_snapshot_on_disk() {
        use crate::llamacpp::prompt_snapshot::SnapshotSlot;
        use llama_cpp_2::model::params::LlamaModelParams;
        use llama_cpp_2::model::AddBos;

        let Some(src) = live_model_path() else {
            eprintln!("skipping: no local GGUF found");
            return;
        };
        let tmp = tempfile::tempdir().expect("tempdir");
        let model_path = tmp.path().join("model.gguf");
        std::fs::hard_link(&src, &model_path).expect("hard link the model, never symlink it");
        // Process-wide, so it is taken under the same lock the registry tests
        // use rather than set directly — two live tests otherwise race and each
        // reads the other's cache directory.
        let _env = env_lock::lock_env([("GOOSE_PATH_ROOT", Some(tmp.path().to_str().unwrap()))]);

        let backend = shared_backend();
        let model = Arc::new(
            LlamaModel::load_from_file(
                backend.llama_backend(),
                &model_path,
                &LlamaModelParams::default(),
            )
            .expect("load model"),
        );
        let settings = ModelSettings::default();
        let n_ctx: usize = 4096;

        // Two prompts of ONE shape: same long preamble, different tails. That is
        // what a second turn of the same caller looks like, and it is the least
        // a snapshot needs in order to know what is stable.
        let preamble = "The pond keeps its own notes about the household. ".repeat(120);
        let tok = |t: &str, bos| model.str_to_token(t, bos).expect("tokenize");
        let base = tok(&preamble, AddBos::Always);
        assert!(base.len() > 1024, "preamble must clear the threshold");

        let first: Vec<LlamaToken> = base
            .iter()
            .chain(tok(" What is the weather?", AddBos::Never).iter())
            .copied()
            .collect();
        let second: Vec<LlamaToken> = base
            .iter()
            .chain(tok(" Remind me about the shed roof.", AddBos::Never).iter())
            .copied()
            .collect();

        // A SECOND shape, sharing nothing with the first — this is the scheduler
        // or the memory extractor beside chat, and it is the case the old hook
        // could never serve.
        let other_preamble =
            "Review the household's recent events and decide whether to speak. ".repeat(90);
        let other_base = tok(&other_preamble, AddBos::Always);
        assert!(
            other_base.len() > 1024,
            "second preamble must clear the threshold"
        );
        let other_first: Vec<LlamaToken> = other_base
            .iter()
            .chain(tok(" Anything worth raising?", AddBos::Never).iter())
            .copied()
            .collect();
        let other_second: Vec<LlamaToken> = other_base
            .iter()
            .chain(tok(" Summarise instead.", AddBos::Never).iter())
            .copied()
            .collect();

        let mut session: Option<SessionKv> = None;
        let mut snapshot: Option<SnapshotSlot> = None;

        // Interleaved, the way a pond actually runs them.
        for prompt in [
            &first,
            &other_first,
            &second,
            &other_second,
            &first,
            &other_first,
        ] {
            prefill_prompt(
                &mut session,
                &mut snapshot,
                &model_path,
                &model,
                backend,
                &settings,
                prompt,
                n_ctx,
            )
            .expect("prefill");
        }

        let slot = snapshot.expect("a slot is built on the first prefill");
        let files = slot.snapshot_paths();
        for f in &files {
            assert!(f.exists(), "the index names {f:?} but it is not there");
        }
        eprintln!("snapshots after six interleaved turns: {}", files.len());
        assert_eq!(
            files.len(),
            2,
            "each prompt shape must get its OWN snapshot. One means only the shape that \
             happened to hold the retained cache could write, which is the defect: a pond \
             runs chat, extraction, titling and the reviewer against one model and they \
             share nothing at the start"
        );
    }

    /// HOW MUCH common ground do two chats actually share?
    ///
    /// The snapshot can only ever cache the run of tokens every chat begins
    /// with, so this measures it against the real template rather than assuming.
    /// The answer decides whether the feature helps chat at all: GIAP's
    /// `tool_selection_mode = "relevant"` picks a different tool set per
    /// conversation — 30 distinct sets across 84 sessions on my own pond — and
    /// the template renders the system text FIRST and the tool declarations
    /// AFTER it. Everything from the first differing tool onward diverges.
    #[test]
    #[ignore = "requires a real GGUF on disk; run with --ignored"]
    fn how_much_two_chats_share_depends_on_whether_their_tools_match() {
        use llama_cpp_2::model::params::LlamaModelParams;

        let Some(src) = live_model_path() else {
            eprintln!("skipping: no local GGUF found");
            return;
        };
        let backend = shared_backend();
        let model =
            LlamaModel::load_from_file(backend.llama_backend(), &src, &LlamaModelParams::default())
                .expect("load model");
        let settings = ModelSettings::default();
        let templates = crate::llamacpp::load_chat_templates(&model, &settings).expect("templates");
        let template = templates
            .tool_use
            .as_ref()
            .or(templates.default.as_ref())
            .expect("a template");

        // A GIAP-sized system prompt: the compact static prefix is capped near
        // 600 tokens, so this is the right order of magnitude.
        let system = "You are the assistant for this household. ".repeat(60);
        let tool = |n: usize| {
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": format!("giap_tool_{n}"),
                    "description": format!(
                        "Tool number {n} for the household, with enough description to be a \
                         realistic schema rather than a token or two."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "target": {"type": "string", "description": "what to act on"},
                            "value": {"type": "string", "description": "the value to use"}
                        },
                        "required": ["target"]
                    }
                }
            })
        };
        let toolset = |ids: &[usize]| {
            serde_json::Value::Array(ids.iter().map(|n| tool(*n)).collect()).to_string()
        };

        let render = |tools: &str| {
            let messages = format!(
                r#"[{{"role":"system","content":{}}},{{"role":"user","content":"hello"}}]"#,
                serde_json::to_string(&system).unwrap()
            );
            let params = OpenAIChatTemplateParams {
                messages_json: &messages,
                tools_json: Some(tools),
                tool_choice: None,
                json_schema: None,
                grammar: None,
                reasoning_format: None,
                chat_template_kwargs: None,
                add_generation_prompt: true,
                use_jinja: true,
                parallel_tool_calls: false,
                enable_thinking: false,
                add_bos: false,
                add_eos: false,
                parse_tool_calls: true,
            };
            let r = model
                .apply_chat_template_oaicompat(template, &params)
                .expect("apply template");
            model
                .str_to_token(&r.prompt, llama_cpp_2::model::AddBos::Always)
                .expect("tokenize")
        };

        let a = render(&toolset(&[1, 2, 3, 4, 5, 6, 7, 8]));
        let same = render(&toolset(&[1, 2, 3, 4, 5, 6, 7, 8]));
        let differs = render(&toolset(&[1, 2, 3, 4, 9, 10, 11, 12]));
        // Same eight-vs-eight difference, but the shared tools are listed FIRST.
        // Order is GIAP's to choose: it builds the tools JSON the template
        // renders, and the always-on extensions are known.
        let differs_late = render(&toolset(&[1, 2, 3, 4, 5, 6, 9, 10]));

        let identical = common_prefix_len(&a, &same);
        let overlapping = common_prefix_len(&a, &differs);
        eprintln!("prompt tokens                : {}", a.len());
        eprintln!(
            "shared, SAME tool set        : {identical} ({:.0}%)",
            100.0 * identical as f64 / a.len() as f64
        );
        eprintln!(
            "shared, DIFFERENT tool set   : {overlapping} ({:.0}%)",
            100.0 * overlapping as f64 / a.len() as f64
        );
        let late = common_prefix_len(&a, &differs_late);
        eprintln!(
            "shared, DIFFERING TOOLS LAST : {late} ({:.0}%)",
            100.0 * late as f64 / a.len() as f64
        );
        eprintln!("snapshot threshold           : {}", 1024);

        assert_eq!(
            identical,
            a.len(),
            "two chats with the same tools must share everything -- if not, something in the \
             preamble varies that the cache can never capture"
        );
        assert!(
            overlapping < identical,
            "a differing tool set has to diverge somewhere, or this measurement is not \
             measuring what it claims"
        );
        assert!(
            late > overlapping,
            "putting the tools two chats SHARE before the ones they do not must lengthen the \
             cacheable run -- this is the lever that makes a disk cache work under \
             tool_selection_mode = relevant, where 30 distinct tool sets across 84 sessions \
             otherwise leave almost nothing in common"
        );
    }

    /// The process's one llama.cpp backend.
    ///
    /// `LlamaCppBackend::new()` treats a second initialisation as `unreachable!`
    /// — correctly, since the runtime holds the only one for the life of the
    /// process — so two live tests each building their own panics the second.
    fn shared_backend() -> &'static LlamaCppBackend {
        static BACKEND: std::sync::OnceLock<LlamaCppBackend> = std::sync::OnceLock::new();
        BACKEND.get_or_init(|| LlamaCppBackend::new().expect("backend"))
    }

    /// The one model this repo ships on-device, if the developer has it.
    fn live_model_path() -> Option<std::path::PathBuf> {
        let home = std::env::var_os("HOME")?;
        let candidates = [
            "Library/Application Support/goose-in-a-pond/models/gguf/gemma-4-E2B-it-Q4_K_M.gguf",
            ".local/share/goose-in-a-pond/models/gguf/gemma-4-E2B-it-Q4_K_M.gguf",
        ];
        candidates
            .iter()
            .map(|rel| std::path::PathBuf::from(&home).join(rel))
            .find(|p| p.is_file())
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
