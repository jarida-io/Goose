use crate::backend::LocalInferenceBackend;
use crate::local_model_registry::ModelSettings;
use crate::multimodal::ExtractedImage;
use goose_provider_types::errors::ProviderError;
use goose_provider_types::request_log::{LoggerHandleExt, RequestLogHandle};
use llama_cpp_2::context::params::{KvCacheType, LlamaContextParams};
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::{AddBos, ChatTemplateResult, LlamaChatTemplate, LlamaModel};
use llama_cpp_2::mtmd::{
    MtmdBitmap, MtmdContext, MtmdInputChunk, MtmdInputChunkType, MtmdInputChunks, MtmdInputText,
};
use llama_cpp_2::openai::OpenAIChatTemplateParams;
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use llama_cpp_2::{LlamaStateSeqFlags, SeqState};
use std::num::NonZeroU32;
use std::sync::Arc;

use super::super::StreamSender;
use super::memory::{KvGeometry, MemInfo, MemoryFit, COMPUTE_RESERVE_BYTES};
use super::LlamaCppBackend;
use crate::load_error::LoadStage;

const MIB: u64 = 1024 * 1024;

/// Shortest shared prompt prefix worth a partial KV cache removal. Below this
/// the bookkeeping costs more than the decode it saves.
const REUSE_MIN_TOKENS: usize = 256;

/// How long a chat's cache stays protected from another chat's prompt. A conversation's turns,
/// side calls and subagents come within seconds; a cache idle longer belongs to one that is over.
const CHAT_CACHE_PROTECTED_FOR: std::time::Duration = std::time::Duration::from_secs(60);

/// Sliding-window checkpoints kept per retained context, newest last. Each is
/// the window layers' cells, 16 MiB for Gemma 4 E4B at q8_0 and 768 cells.
const WINDOW_CHECKPOINTS: usize = 4;

/// How far behind the end of a prefill window checkpoints reach, one at each
/// batch boundary from the last one at or before the start of the span. A
/// turn's prompt usually diverges from the cache within the last turn: the
/// reply as re-rendered, the new message and its context. With the Orin's batch
/// of 512 that is at most three, and the checkpoint at the end makes four.
const WINDOW_CHECKPOINT_SPAN: usize = 1024;

/// The sliding-window layers' cells with the cache holding positions `[0, pos)`.
///
/// A windowed cache keeps only the last `n_swa` positions plus a ubatch of
/// slack in those layers, so it can resume only a short way behind its end; a
/// checkpoint is how a deeper rewind still resumes near where it lands.
struct WindowCheckpoint {
    pos: usize,
    /// Fingerprint of the tokens at `[0, pos)`. A checkpoint is restored only
    /// under a prompt with the same prefix, whatever rewound the cache since.
    prefix: u64,
    state: SeqState,
}

/// FNV-1a over token ids.
fn token_fingerprint(tokens: &[LlamaToken]) -> u64 {
    tokens.iter().fold(0xcbf2_9ce4_8422_2325, |hash, token| {
        token.0.to_le_bytes().iter().fold(hash, |h, byte| {
            (h ^ u64::from(*byte)).wrapping_mul(0x100_0000_01b3)
        })
    })
}

/// The model's attention window for its sliding-window layers; 0 when it has
/// none, as the GGUF says (`<arch>.attention.sliding_window`).
fn sliding_window(model: &LlamaModel) -> usize {
    let Ok(arch) = model.meta_val_str("general.architecture") else {
        return 0;
    };
    model
        .meta_val_str(&format!("{arch}.attention.sliding_window"))
        .ok()
        .and_then(|value| value.trim().parse().ok())
        .unwrap_or(0)
}

pub(super) struct GenerationContext<'a> {
    pub model: &'a Arc<LlamaModel>,
    /// MTP drafter for this slot. `None` means no speculation.
    pub draft: Option<&'a Arc<LlamaModel>>,
    pub mtmd_ctx: Option<&'a MtmdContext>,
    pub session: &'a mut Option<SessionKv>,
    /// On-disk preamble cache for this model. `None` disables snapshots for the
    /// request without changing any other behaviour.
    pub snapshot: &'a mut Option<super::prompt_snapshot::SnapshotSlot>,
    /// The GGUF this slot loaded, for keying that cache.
    pub model_path: &'a std::path::Path,
    /// The context budget fixed when the model loaded; `None` off unified memory.
    pub memory_fit: Option<MemoryFit>,
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
    /// The request carries no tools: one-shot work (a compaction's summary,
    /// memory extraction, a title) rather than a turn of a chat.
    pub one_shot: bool,
    /// Only a tool call from this generation is used (`LocalGenerationRequest::tool_calls_only`).
    pub tool_calls_only: bool,
    /// A helper's call (`LocalGenerationRequest::helper`).
    pub helper: bool,
    /// A live turn waits on this helper (`LocalGenerationRequest::prime_waiting`).
    pub prime_waiting: bool,
    /// The waiting turn's cache, set aside while a helper holds the context.
    pub suspended: &'a mut Option<SuspendedKv>,
    /// The shape a prefill compile names (`LocalGenerationRequest::compile_prefill`).
    pub compile_prefill: Option<String>,
    /// How long a helper snapshot may go unused before a compile discards it.
    pub prefill_retention_secs: Option<u64>,
    /// The compile rebuilds even a valid snapshot (`LocalGenerationRequest::rebuild_prefill`).
    pub rebuild_prefill: bool,
}

pub(super) struct LoadedModel {
    pub model: Arc<LlamaModel>,
    /// MTP drafter, loaded beside the target when the registry names one.
    /// `None` disables speculation entirely -- there is no partial mode.
    pub draft: Option<Arc<LlamaModel>>,
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
    /// A waiting turn's cache on disk while a helper holds `session`; dropping it deletes the file.
    pub suspended: Option<SuspendedKv>,
    /// Where this model was loaded from. Part of the snapshot key.
    pub model_path: std::path::PathBuf,
    /// What a context may spend on a unified-memory board, measured once after the load.
    pub memory_fit: Option<MemoryFit>,
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
/// Either a bare context, or a speculative wrapper that has TAKEN it.
///
/// `MtpSpeculative::new` consumes both the target and draft contexts by value, so
/// a session that speculates cannot also hold its context directly. Everything
/// that used `kv.ctx` reaches the same context through [`SessionCtx::ctx_mut`],
/// which keeps `reuse_prefix`, the snapshot ladder and `clear_kv_cache_seq`
/// working unchanged -- the prompt-session cache is worth more than speculation
/// and must not be traded for it.
pub(super) enum SessionCtx {
    /// No drafter configured.
    Plain(LlamaContext<'static>),
    /// Drafter attached; the target context lives inside.
    Mtp(Box<crate::llamacpp::mtp::MtpSession>),
}

impl SessionCtx {
    pub(super) fn ctx(&self) -> &LlamaContext<'static> {
        match self {
            Self::Plain(c) => c,
            Self::Mtp(m) => m.target(),
        }
    }

    pub(super) fn ctx_mut(&mut self) -> &mut LlamaContext<'static> {
        match self {
            Self::Plain(c) => c,
            Self::Mtp(m) => m.target_mut(),
        }
    }

    pub(super) fn mtp_mut(&mut self) -> Option<&mut crate::llamacpp::mtp::MtpSession> {
        match self {
            Self::Plain(_) => None,
            Self::Mtp(m) => Some(m),
        }
    }
}

pub(super) struct SessionKv {
    /// Declared before `_model`: fields drop in declaration order, so the
    /// context is destroyed before the model allocation it points into.
    ctx: SessionCtx,
    head: Option<MediaHead>,
    tokens: Vec<LlamaToken>,
    /// The `n_ctx` this context was *requested* with. Compared against the
    /// effective context of an incoming request; llama.cpp may round the
    /// realised `n_ctx` up, so the requested value is the stable identity.
    n_ctx: u32,
    /// The last prefill into this cache was one-shot work or a helper's, so it protects nothing:
    /// at most that helper's own next turn resumes from it (see `prefill_plan`).
    one_shot: bool,
    /// When a generation last ran on this cache.
    last_used: std::time::Instant,
    /// The model's sliding window, 0 for a model without one.
    n_swa: usize,
    checkpoints: Vec<WindowCheckpoint>,
    _model: Arc<LlamaModel>,
    /// Same drop-order contract as `_model`: the speculative wrapper inside
    /// `ctx` borrows this allocation, so it must outlive the context.
    _draft_model: Option<Arc<LlamaModel>>,
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
    /// Create a context of `n_ctx`. A failure is retried once after releasing the loaded
    /// models' files from the page cache (logged as `kind="model_load_retry"`), and a second
    /// one with free memory short of the KV cache is
    /// [`OutOfGpuMemory`](crate::load_error::ModelLoadError::OutOfGpuMemory).
    pub(super) fn create(
        model: &Arc<LlamaModel>,
        draft: Option<&Arc<LlamaModel>>,
        backend: &LlamaCppBackend,
        n_ctx: u32,
        settings: &ModelSettings,
    ) -> Result<Self, ProviderError> {
        let first = match Self::try_create(model, draft, backend, n_ctx, settings) {
            Ok(kv) => return Ok(kv),
            Err(e) if backend.retries_after_release() => e,
            Err(e) => return Err(e),
        };
        let before = MemInfo::read();
        let after = backend.release_model_files("context_retry");
        tracing::warn!(
            target: "giap::trace",
            kind = "model_load_retry",
            stage = "context",
            n_ctx,
            error = %first,
            free_mb_before = before.map(|m| m.free / MIB),
            free_mb_after = after.map(|m| m.free / MIB),
            cached_mb_before = before.map(|m| m.cached / MIB),
            cached_mb_after = after.map(|m| m.cached / MIB),
            "context creation failed; retrying once with the model files released from the page cache"
        );
        Self::try_create(model, draft, backend, n_ctx, settings).map_err(|e| {
            let type_v = settings
                .type_v
                .as_deref()
                .filter(|_| settings.flash_attention == Some(true));
            let kv_bytes = KvGeometry::of(model).map_or(0, |g| {
                u64::from(n_ctx)
                    .saturating_mul(g.bytes_per_token(settings.type_k.as_deref(), type_v))
            });
            let needed_mb = kv_bytes.saturating_add(COMPUTE_RESERVE_BYTES) / MIB;
            backend.classify_failure(LoadStage::Context, needed_mb, e.to_string())
        })
    }

    fn try_create(
        model: &Arc<LlamaModel>,
        draft: Option<&Arc<LlamaModel>>,
        backend: &LlamaCppBackend,
        n_ctx: u32,
        settings: &ModelSettings,
    ) -> Result<Self, ProviderError> {
        let model = Arc::clone(model);
        let outputs = outputs_per_ubatch(draft.is_some(), settings);
        let ctx = model
            .new_context(
                backend.llama_backend(),
                build_context_params(n_ctx, settings, outputs),
            )
            .map_err(|e| ProviderError::ExecutionError(format!("Failed to create context: {e}")))?;

        // SAFETY: `ctx` borrows the `LlamaModel` owned by the `Arc` allocation,
        // whose address is stable for as long as any strong handle exists.
        // Erasing the lifetime is sound because it is immediately re-tied to
        // `_model`, a strong handle stored in this same struct and dropped
        // *after* `ctx` by field declaration order. The context can therefore
        // never outlive the model it points at.
        let ctx = unsafe { std::mem::transmute::<LlamaContext<'_>, LlamaContext<'static>>(ctx) };

        // Attach the drafter, if the registry named one. The draft context gets
        // the same window as the target: llama.cpp's MTP drafter shares the
        // target's KV rather than duplicating it, which is why the whole
        // configuration measured +48 MB on the Orin rather than a second cache.
        let (ctx, draft_model) = match draft {
            Some(draft_model) => {
                let draft_model = Arc::clone(draft_model);
                // The draft context MUST reference the target's via llama.cpp's
                // `ctx_other`. Gemma4Assistant refuses to build without it --
                // "requires ctx_other to be set" -- and it is also how the KV is
                // shared: `is_mem_shared = llama_get_ctx_other(ctx_dft) ==
                // ctx_tgt`, which is why the whole configuration measured +48 MB
                // on the Orin rather than a second full cache. Building it as a
                // plain context returns a null and fails the load.
                let dctx = draft_model
                    .new_context_with_ctx_other(
                        backend.llama_backend(),
                        build_context_params(n_ctx, settings, outputs),
                        &ctx,
                    )
                    .map_err(|e| {
                        ProviderError::ExecutionError(format!(
                            "failed to create draft context: {e}"
                        ))
                    })?;
                // SAFETY: as for the target context above -- the borrow is
                // immediately re-tied to `_draft_model`, a strong handle in this
                // same struct dropped after `ctx` by declaration order.
                let dctx =
                    unsafe { std::mem::transmute::<LlamaContext<'_>, LlamaContext<'static>>(dctx) };
                // Depth 4 because that is what the Orin bake-off measured:
                // 2.8x at 87% acceptance. Do not raise it on the strength of a
                // Mac run -- ggml-metal switches from mat-vec to GEMM only above
                // a batch of 8, so on Metal depth 4 lands in the one place
                // batching does not amortise and depth 8 looks better. That is a
                // property of the backend, not of the drafter, and the device
                // this ships on does not share it.
                let session = super::mtp::MtpSession::new(
                    ctx,
                    dctx,
                    settings.draft_n_max.unwrap_or(4),
                    settings.draft_p_min.unwrap_or(0.0),
                )?;
                (SessionCtx::Mtp(Box::new(session)), Some(draft_model))
            }
            None => (SessionCtx::Plain(ctx), None),
        };

        let n_swa = sliding_window(&model);
        Ok(Self {
            ctx,
            head: None,
            tokens: Vec::new(),
            n_ctx,
            one_shot: false,
            last_used: std::time::Instant::now(),
            n_swa,
            checkpoints: Vec::new(),
            _model: model,
            _draft_model: draft_model,
        })
    }

    pub(super) fn context_mut(&mut self) -> &mut LlamaContext<'static> {
        self.ctx.ctx_mut()
    }

    /// The context AND whatever drafter is attached to it.
    pub(super) fn session_ctx_mut(&mut self) -> &mut SessionCtx {
        &mut self.ctx
    }

    pub(super) fn record_generated(&mut self, tokens: &[LlamaToken]) {
        self.tokens.extend_from_slice(tokens);
        self.last_used = std::time::Instant::now();
    }

    /// First position `tokens` describes.
    fn media_pos(&self) -> usize {
        self.head.as_ref().map_or(0, |head| head.n_pos)
    }

    /// Every position the cache is known to hold, reusable or not.
    fn occupied(&self) -> usize {
        self.media_pos() + self.tokens.len()
    }

    /// Saves the window layers' cells while the cache holds exactly `tokens`.
    /// Text-only caches: behind a media head, positions are not token indices.
    fn checkpoint_window(&mut self, tokens: &[LlamaToken]) {
        if self.n_swa == 0 || self.head.is_some() {
            return;
        }
        let pos = tokens.len();
        if self.checkpoints.last().is_some_and(|c| c.pos == pos) {
            return;
        }
        match self
            .ctx
            .ctx()
            .state_seq_get(0, LlamaStateSeqFlags::PARTIAL_ONLY)
        {
            Ok(state) => {
                self.checkpoints.retain(|c| c.pos < pos);
                self.checkpoints.push(WindowCheckpoint {
                    pos,
                    prefix: token_fingerprint(tokens),
                    state,
                });
                if self.checkpoints.len() > WINDOW_CHECKPOINTS {
                    self.checkpoints.remove(0);
                }
            }
            Err(error) => tracing::debug!(%error, pos, "window checkpoint not taken"),
        }
    }

    /// Where a resume at `resume_pos` under `prompt` can start.
    ///
    /// A full-size cache resumes anywhere. A windowed one resumes in place only
    /// while its window layers still hold the `n_swa` positions behind
    /// `resume_pos`; past that, the newest checkpoint at or before it with the
    /// same token prefix puts those cells back, and decoding starts there.
    /// `None`: nothing to resume from.
    fn window_resume(&mut self, prompt: &[LlamaToken], resume_pos: usize) -> Option<usize> {
        if self.n_swa == 0 {
            return Some(resume_pos);
        }
        let pos_min = self.ctx.ctx().kv_cache_seq_pos_min(0);
        if pos_min >= 0 && pos_min as usize <= resume_pos.saturating_sub(self.n_swa) {
            return Some(resume_pos);
        }
        if self.head.is_some() {
            return None;
        }
        let found = self.checkpoints.iter().rposition(|c| {
            c.pos <= resume_pos
                && c.pos <= prompt.len()
                && c.prefix == token_fingerprint(&prompt[..c.pos])
        })?;
        let checkpoint = &self.checkpoints[found];
        match self.ctx.ctx_mut().state_seq_set(&checkpoint.state, 0) {
            Ok(()) => Some(checkpoint.pos),
            Err(error) => {
                tracing::warn!(%error, pos = checkpoint.pos, "window checkpoint did not restore");
                self.checkpoints.clear();
                None
            }
        }
    }
}

/// Context-window size of a retained KV cache, or 0 when nothing is retained.
fn retained_kv_tokens(session: &Option<SessionKv>) -> usize {
    session.as_ref().map_or(0, |kv| kv.n_ctx as usize)
}

pub(super) struct PreparedGeneration {
    pub template_result: ChatTemplateResult,
    /// The prompt as tokens. Carried because a speculative draft call is given
    /// the whole sequence the target stands on, not just its length.
    pub prompt_tokens: Vec<LlamaToken>,
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
    /// Whether a chat filled the cache, rather than one-shot work nothing
    /// resumes from.
    worth_protecting: bool,
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
            worth_protecting: !kv.one_shot,
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
            worth_protecting: !kv.one_shot,
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

/// Whether a cache a chat filled still protects itself from `prompt_is_chat`. A side call never
/// evicts it; another chat does once it has sat idle, or that chat's next turn prefills it all again.
fn chat_cache_protected(prompt_is_chat: bool, idle: std::time::Duration) -> bool {
    !prompt_is_chat || idle < CHAT_CACHE_PROTECTED_FOR
}

/// A chat's cache set aside on disk while a helper holds the model's one context and the chat's
/// turn waits on it. The file holds conversation, so dropping this deletes it.
pub(super) struct SuspendedKv {
    path: std::path::PathBuf,
    tokens: Vec<LlamaToken>,
    n_ctx: u32,
    checkpoints: Vec<WindowCheckpoint>,
}

impl Drop for SuspendedKv {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

const SUSPEND_DIR: &str = "prompt-cache/suspended";

static SUSPEND_LEFTOVERS_CLEARED: std::sync::Once = std::sync::Once::new();

/// Deletes suspend files a crashed process left; once per process, before any is written.
pub(super) fn clear_suspended_leftovers() {
    SUSPEND_LEFTOVERS_CLEARED.call_once(|| {
        let _ = std::fs::remove_dir_all(crate::paths::Paths::in_data_dir(SUSPEND_DIR));
    });
}

/// The suspend directory, readable by the owner alone.
fn suspend_dir() -> std::io::Result<std::path::PathBuf> {
    clear_suspended_leftovers();
    let dir = crate::paths::Paths::in_data_dir(SUSPEND_DIR);
    std::fs::create_dir_all(&dir)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700))?;
    }
    Ok(dir)
}

/// Set the retained chat cache aside on disk before a helper takes the context. `None` leaves it
/// in place, and the helper then evicts it as it would with no turn waiting.
fn suspend_for_helper(session: &mut Option<SessionKv>) -> Option<SuspendedKv> {
    let kv = session.as_ref()?;
    // One-shot work and a picture's head are never resumed from, so they are not worth a file.
    if kv.one_shot || kv.head.is_some() || kv.tokens.len() < REUSE_MIN_TOKENS {
        return None;
    }
    let started = std::time::Instant::now();
    let path = match suspend_dir() {
        Ok(dir) => dir.join(format!("{}.kv", uuid::Uuid::new_v4())),
        Err(e) => {
            tracing::warn!(target: "giap::kv", event = "not_written", error = %e, "suspend directory unavailable; the helper evicts the chat cache");
            return None;
        }
    };
    let written = kv
        .ctx
        .ctx()
        .state_seq_save_file(&path, 0, &kv.tokens)
        .map_err(|e| e.to_string())
        .and_then(|bytes| restrict_to_owner(&path).map(|()| bytes));
    let bytes = match written {
        Ok(bytes) => bytes,
        Err(e) => {
            let _ = std::fs::remove_file(&path);
            tracing::warn!(target: "giap::kv", event = "not_written", error = %e, "chat cache not suspended; the helper evicts it");
            return None;
        }
    };
    let SessionKv {
        tokens,
        n_ctx,
        checkpoints,
        ..
    } = session.take()?;
    tracing::info!(
        target: "giap::kv",
        event = "suspended",
        tokens = tokens.len(),
        bytes,
        elapsed_ms = started.elapsed().as_millis() as u64,
        "chat cache set aside on disk while a helper holds the context"
    );
    Some(SuspendedKv {
        path,
        tokens,
        n_ctx,
        checkpoints,
    })
}

fn restrict_to_owner(path: &std::path::Path) -> Result<(), String> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
            .map_err(|e| e.to_string())?;
    }
    let _ = path;
    Ok(())
}

/// Bring a suspended chat cache back for the prompt that continues it, or delete it when this
/// prompt does not. Either way the file is gone afterwards.
#[allow(clippy::too_many_arguments)]
fn resume_suspended(
    session: &mut Option<SessionKv>,
    suspended: &mut Option<SuspendedKv>,
    draft: Option<&Arc<LlamaModel>>,
    model: &Arc<LlamaModel>,
    backend: &LlamaCppBackend,
    settings: &ModelSettings,
    prompt: &[LlamaToken],
    n_ctx: u32,
) {
    let Some(mut held) = suspended.take() else {
        return;
    };
    if held.n_ctx != n_ctx || common_prefix_len(&held.tokens, prompt) < REUSE_MIN_TOKENS {
        tracing::info!(
            target: "giap::kv",
            event = "discarded",
            reason = "the next chat prompt does not continue the suspended one",
            "suspended chat cache deleted"
        );
        return;
    }
    let started = std::time::Instant::now();
    // Released first, so two caches are never resident at once.
    *session = None;
    let mut kv = match SessionKv::create(model, draft, backend, n_ctx, settings) {
        Ok(kv) => kv,
        Err(e) => {
            tracing::warn!(target: "giap::kv", event = "discarded", error = %e, "no context to resume the suspended chat cache into");
            return;
        }
    };
    match kv
        .context_mut()
        .state_seq_load_file(&held.path, 0, n_ctx as usize)
    {
        Ok((tokens, bytes)) if tokens == held.tokens => {
            kv.tokens = std::mem::take(&mut held.tokens);
            kv.checkpoints = std::mem::take(&mut held.checkpoints);
            tracing::info!(
                target: "giap::kv",
                event = "resumed",
                tokens = kv.tokens.len(),
                bytes,
                elapsed_ms = started.elapsed().as_millis() as u64,
                "suspended chat cache resumed; the turn continues from it"
            );
            *session = Some(kv);
        }
        Ok(_) => {
            tracing::warn!(target: "giap::kv", event = "discarded", reason = "the file held other tokens", "suspended chat cache not resumed");
        }
        Err(e) => {
            tracing::warn!(target: "giap::kv", event = "discarded", error = %e, "suspended chat cache could not be read");
        }
    }
}

/// Whether the retained cache is shielded from this prompt. A helper takes the one context: a
/// second beside the prime's does not fit the device.
fn retained_cache_protected(helper: bool, prompt_is_chat: bool, idle: std::time::Duration) -> bool {
    !helper && chat_cache_protected(prompt_is_chat, idle)
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
    // Only a cache a chat filled is worth protecting. One-shot work leaves a
    // cache nothing resumes from: on a Jetson Orin, a compaction summary stopped
    // halfway left 13K tokens of summary prompt, and a new conversation's turns
    // then ran in throwaway contexts, each a full prefill (7-8 s to the first
    // token), until the conversation outgrew half of it.
    if cached.worth_protecting && is_sacrificial_prompt(prompt.len(), cached.occupied) {
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

/// The window a request gets: the requested one ([`requested_context`]), narrowed to what the
/// memory free at load buys on a unified-memory board ([`MemoryFit::fit`]). The fit only ever
/// narrows, so a pinned window or a host's `context_window_override` that already fits is kept
/// exactly; one that does not fit would fail to allocate rather than run.
pub(super) fn context_cap(
    settings: &crate::local_model_registry::ModelSettings,
    context_limit: usize,
    n_ctx_train: usize,
    memory_max_ctx: Option<usize>,
    memory_fit: Option<MemoryFit>,
) -> usize {
    let requested = requested_context(settings, context_limit, n_ctx_train, memory_max_ctx);
    let Some(fit) = memory_fit else {
        return requested;
    };
    let chosen = fit.fit(requested);
    if chosen.n_ctx != requested {
        tracing::debug!(
            target: "giap::trace",
            requested,
            n_ctx = chosen.n_ctx,
            reason = chosen.reason.as_str(),
            "context window narrowed to the memory free at load"
        );
    }
    chosen.n_ctx
}

/// The window the settings, the request and the engine's own estimate ask for, before memory.
fn requested_context(
    settings: &crate::local_model_registry::ModelSettings,
    context_limit: usize,
    n_ctx_train: usize,
    memory_max_ctx: Option<usize>,
) -> usize {
    // 1. Explicit context_size in model settings (highest priority)
    //
    // Clamped to the trained window like every other branch below. It was the
    // one branch that was not, which made it possible to pin a context larger
    // than the model was trained for and get degraded output with no error and
    // no log line. Latent for the models GIAP ships (both train past its
    // MAX_CTX of 16384) but not for the MTP drafter, which is handed the
    // target's window by construction and whose own trained window is never
    // compared against it.
    //
    // A host that genuinely owns the memory budget still wins here — the clamp
    // only refuses a window the weights cannot support, which is never the
    // budget's decision to make.
    if let Some(ctx_size) = settings.context_size {
        let requested = ctx_size as usize;
        // `n_ctx_train == 0` means the header did not say; clamping to it would
        // return a zero-length context, which is worse than not clamping.
        let granted = if n_ctx_train > 0 {
            requested.min(n_ctx_train)
        } else {
            requested
        };
        if granted != requested {
            tracing::warn!(
                target: "giap::kv",
                requested,
                n_ctx_train,
                granted,
                "pinned context_size exceeds the model's trained window; clamping",
            );
        }
        return granted;
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
    memory_fit: Option<MemoryFit>,
) -> usize {
    let limit = context_cap(
        settings,
        context_limit,
        n_ctx_train,
        memory_max_ctx,
        memory_fit,
    );
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

/// LOAD-BEARING FOR KV REUSE: contexts keep a windowed cache (`swa_full =
/// false`), and every partial resume goes through [`SessionKv::window_resume`].
///
/// A sliding-window model (Gemma among them) needs, to decode at a position,
/// the `n_swa` positions behind it in each sliding-window layer. A windowed
/// cache keeps only the last `n_swa` positions plus a ubatch there, so a resume
/// further back than that would attend to KV that is no longer there and
/// quietly produce different output -- no error, just a worse answer.
/// `window_resume` resumes in place only when the window behind the resume
/// position is still held, restores a window checkpoint (with the same token
/// prefix) otherwise, and refuses when there is none. Every resume path must
/// keep going through it.
///
/// Why not the full-size window cache: on a Jetson Orin, Gemma 4 E4B at 16K
/// context kept 476 MiB of KV with it and 152 MiB without (q8_0), and decoded
/// 8% slower at 4,400 tokens of context, its sliding-window layers attending
/// over every cached position rather than the last 512.
pub(super) fn build_context_params(
    ctx_size: u32,
    settings: &crate::local_model_registry::ModelSettings,
    n_outputs_max: u32,
) -> LlamaContextParams {
    // Left at its default, llama.cpp reserves a 262,144-wide f32 row of logits per ubatch token
    // for Gemma: 128 MiB at n_ubatch 128, 512 MiB at 512. See `outputs_per_ubatch`.
    let mut params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(ctx_size))
        .with_swa_full(false)
        .with_n_outputs_max(n_outputs_max);

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
    if let Some(n_ubatch) = settings.n_ubatch {
        params = params.with_n_ubatch(n_ubatch);
    }
    if let Some(name) = settings.type_k.as_deref() {
        match parse_kv_cache_type(name) {
            Some(t) => params = params.with_type_k(t),
            None => tracing::warn!("Unknown type_k '{}'; leaving KV K at f16", name),
        }
    }
    if let Some(name) = settings.type_v.as_deref() {
        // llama.cpp rejects a quantised V cache without flash attention. Refuse
        // here with a reason rather than letting context creation fail opaquely.
        if settings.flash_attention != Some(true) {
            tracing::warn!(
                "type_v '{}' ignored: quantising the V cache requires flash_attention = true",
                name
            );
        } else {
            match parse_kv_cache_type(name) {
                Some(t) => params = params.with_type_v(t),
                None => tracing::warn!("Unknown type_v '{}'; leaving KV V at f16", name),
            }
        }
    }

    params
}

/// The most logits one ubatch asks for. A plain context asks for the last token's only
/// (`decode_tokens`, the generation loop, mtmd's `logits_last`). With a drafter attached, the
/// target verifies the last token and every draft at once, and llama.cpp aborts a decode that asks
/// for more outputs than the context reserved.
fn outputs_per_ubatch(
    speculative: bool,
    settings: &crate::local_model_registry::ModelSettings,
) -> u32 {
    if speculative {
        settings.draft_n_max.unwrap_or(4).max(0) as u32 + 1
    } else {
        1
    }
}

/// Map a ggml type name onto [`KvCacheType`].
///
/// Only the types that are sane for a KV cache are accepted; `q8_0` is the one
/// measured to be quality-neutral on gemma-4 (paired per-chunk wikitext-2,
/// n = 100: dNLL -0.000987 +/- 0.000551, t = -1.79, i.e. indistinguishable from
/// f16, and byte-identical greedy output). `q4_0` measured t = 0.35 on the mean
/// but with 6.4x the per-chunk variance, so it is available and not recommended.
fn parse_kv_cache_type(name: &str) -> Option<KvCacheType> {
    match name.to_ascii_lowercase().as_str() {
        "f32" => Some(KvCacheType::F32),
        "f16" => Some(KvCacheType::F16),
        "bf16" => Some(KvCacheType::BF16),
        "q8_0" => Some(KvCacheType::Q8_0),
        "q5_1" => Some(KvCacheType::Q5_1),
        "q5_0" => Some(KvCacheType::Q5_0),
        "q4_1" => Some(KvCacheType::Q4_1),
        "q4_0" => Some(KvCacheType::Q4_0),
        "iq4_nl" => Some(KvCacheType::IQ4_NL),
        _ => None,
    }
}

pub(super) fn build_sampler(
    model: &LlamaModel,
    settings: &crate::local_model_registry::ModelSettings,
) -> LlamaSampler {
    use crate::local_model_registry::SamplingConfig;

    let has_penalties = settings.repeat_penalty != 1.0
        || settings.frequency_penalty != 0.0
        || settings.presence_penalty != 0.0;

    let mut samplers: Vec<LlamaSampler> = Vec::new();

    if has_penalties {
        // 0.1.156 added `n_vocab` as the first parameter (it was implicit before).
        samplers.push(LlamaSampler::penalties(
            model.n_vocab(),
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
// Eight: the load-time memory fit joined the seven inputs the window already had.
#[allow(clippy::too_many_arguments)]
pub(super) fn validate_and_compute_context(
    model: &LlamaModel,
    has_mtmd: bool,
    backend: &LlamaCppBackend,
    prompt_token_count: usize,
    context_limit: usize,
    settings: &crate::local_model_registry::ModelSettings,
    retained_kv_tokens: usize,
    memory_fit: Option<MemoryFit>,
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
        memory_fit,
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

/// Decode `prompt[from..]` at positions `from..` into a text-only cache,
/// checkpointing the window layers before each batch that ends inside the last
/// [`WINDOW_CHECKPOINT_SPAN`] tokens, so a later rewind into this stretch
/// resumes near where it lands. Behind a media head, or for a model without a
/// sliding window, this is [`decode_tokens`].
fn decode_checkpointed(
    kv: &mut SessionKv,
    prompt: &[LlamaToken],
    from: usize,
) -> Result<(), ProviderError> {
    if kv.n_swa == 0 || kv.head.is_some() {
        return decode_tokens(kv.ctx.ctx_mut(), &prompt[from..], from);
    }
    let n_batch = (kv.ctx.ctx().n_batch() as usize).max(1);
    let span_start = prompt.len().saturating_sub(WINDOW_CHECKPOINT_SPAN);
    let mut pos = from;
    while pos < prompt.len() {
        let end = (pos + n_batch).min(prompt.len());
        if pos > from && end > span_start {
            kv.checkpoint_window(&prompt[..pos]);
        }
        decode_tokens(kv.ctx.ctx_mut(), &prompt[pos..end], pos)?;
        pos = end;
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
/// With a windowed cache the decode may start earlier than `resume_pos`, at a
/// window checkpoint (see [`SessionKv::window_resume`]); returns the position
/// it started at.
fn reuse_prefix(
    kv: &mut SessionKv,
    prompt: &[LlamaToken],
    resume_pos: usize,
) -> Result<usize, PrefixReuseError> {
    let resume_pos = kv
        .window_resume(prompt, resume_pos)
        .ok_or(PrefixReuseError::Refused)?;
    let kept = kept_prefix_tokens(resume_pos, kv.media_pos(), prompt.len(), kv.tokens.len())
        .ok_or(PrefixReuseError::Refused)?;
    let p0 = u32::try_from(resume_pos).map_err(|_| PrefixReuseError::Refused)?;
    match kv.ctx.ctx_mut().clear_kv_cache_seq(Some(0), Some(p0), None) {
        Ok(true) => {}
        Ok(false) | Err(_) => return Err(PrefixReuseError::Refused),
    }
    kv.checkpoints.retain(|c| c.pos <= resume_pos);

    kv.tokens.truncate(kept);
    if kv.head.is_some() {
        decode_tokens(kv.ctx.ctx_mut(), &prompt[kept..], resume_pos)
    } else {
        decode_checkpointed(kv, prompt, kept)
    }
    .map_err(PrefixReuseError::Poisoned)?;
    kv.tokens.extend_from_slice(&prompt[kept..]);
    Ok(resume_pos)
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
    match kv.ctx.ctx_mut().clear_kv_cache_seq(Some(0), Some(p0), None) {
        Ok(true) => {
            kv.tokens.truncate(keep);
            kv.checkpoints.retain(|c| c.pos <= keep);
            Ok(())
        }
        Ok(false) | Err(_) => Err(()),
    }
}

/// The prompt prefilled on top of the preamble snapshot, when one matches.
// Nine arguments: the session, the snapshot slot, the model, its drafter and
// the backend that own them, the settings, the prompt and both windows.
#[allow(clippy::too_many_arguments)]
fn restore_preamble(
    session: &mut Option<SessionKv>,
    snapshot: &mut Option<super::prompt_snapshot::SnapshotSlot>,
    draft: Option<&Arc<LlamaModel>>,
    model: &Arc<LlamaModel>,
    backend: &LlamaCppBackend,
    settings: &ModelSettings,
    prompt: &[LlamaToken],
    n_ctx: u32,
    effective_ctx: usize,
) -> Option<PrefilledPrompt> {
    let slot = snapshot.as_mut()?;
    match seed_from_snapshot(
        session, draft, model, backend, settings, slot, prompt, n_ctx,
    ) {
        Ok(Some(restored)) => Some(PrefilledPrompt {
            reused_prefix_tokens: restored,
            transient: None,
            effective_ctx,
        }),
        // No usable file. `session` now holds a fresh empty context that
        // `full_prefill` decodes into rather than reallocating.
        Ok(None) => None,
        Err(e) => {
            tracing::debug!(error = %e, "snapshot restore failed; prefilling normally");
            None
        }
    }
}

/// Build a fresh context and try to fill its preamble from disk.
///
/// `Ok(Some(n))` means the context now holds the whole prompt with `n` leading
/// positions restored rather than decoded. `Ok(None)` means the context is
/// fresh and empty and the caller should prefill it as usual -- the context is
/// left in `session` either way, so no allocation is wasted.
// Eight because a drafter has to be threaded through beside the model it
// drafts for; bundling them would only move the same list behind a name.
#[allow(clippy::too_many_arguments)]
fn seed_from_snapshot(
    session: &mut Option<SessionKv>,
    draft: Option<&Arc<LlamaModel>>,
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
    let mut kv = SessionKv::create(model, draft, backend, n_ctx, settings)?;

    let Some(restored) = slot.load(kv.context_mut(), prompt) else {
        *session = Some(kv);
        return Ok(None);
    };

    kv.tokens.extend_from_slice(&prompt[..restored]);
    match decode_checkpointed(&mut kv, prompt, restored) {
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
    draft: Option<&Arc<LlamaModel>>,
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
        *session = Some(SessionKv::create(model, draft, backend, n_ctx, settings)?);
    }

    let kv = session
        .as_mut()
        .expect("session was just populated or already matched n_ctx");
    kv.head = None;
    kv.tokens.clear();
    kv.checkpoints.clear();
    kv.ctx.ctx_mut().clear_kv_cache();
    match decode_checkpointed(kv, prompt, 0) {
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
    turn_was_cold: bool,
    class: super::prompt_snapshot::SnapshotClass,
    // A compile's shape and preamble, which a shared bucket's intersection may have shrunk below.
    compiled: Option<(&str, usize)>,
) {
    // Writing is not free: measured 21.2 MiB in 314 ms, about 67 MiB/s, which is
    // ~1.6 s for a 6,000-token preamble on an 18 KiB/token model and ~5 s on a
    // 56 KiB/token one. On a warm turn that finishes in two seconds, paying it
    // would make the cache the slowest thing about the turn it exists to speed
    // up. A cold turn is already spending tens of seconds on prefill and can
    // absorb it -- and a cold turn is exactly when a shape is being established,
    // so nothing is lost by waiting for one.
    if !turn_was_cold {
        return;
    }
    let (Some(slot), Some(kv)) = (snapshot.as_mut(), session.as_mut()) else {
        return;
    };
    // Text-only: with a media head `tokens` starts at `media_pos`, so a token
    // count is not an absolute position.
    if kv.head.is_some() {
        return;
    }
    let Some(keep) = compiled
        .map(|(_, keep)| keep)
        .or_else(|| slot.writable_prefix(prompt))
    else {
        return;
    };
    // The cache has to hold what is about to be written, and there has to be a
    // tail worth keeping the rest of.
    if keep >= kv.tokens.len() || common_prefix_len(&kv.tokens, prompt) < keep {
        return;
    }
    // A windowed cache holds the window behind `keep` only near its end; a
    // checkpoint at or before `keep` puts it back, and the stretch from there
    // to `keep` is decoded again before the write.
    let Some(at) = kv.window_resume(prompt, keep) else {
        // Logged: a shape that is due a snapshot and never gets one is otherwise invisible.
        tracing::info!(
            target: "giap::kv",
            event = "not_written",
            reason = "no window checkpoint at or before the stable prefix",
            stable_tokens = keep,
            prompt_tokens = prompt.len(),
            "kv snapshot not written"
        );
        return;
    };
    if trim_to_prefix(kv, at).is_err() {
        return;
    }
    if at < keep {
        if let Err(e) = decode_tokens(kv.ctx.ctx_mut(), &prompt[at..keep], at) {
            tracing::warn!(
                target: "giap::kv",
                event = "discarded",
                reason = "decoding up to the preamble failed",
                error = %e,
                "kv snapshot not written; rebuilding the context"
            );
            *session = None;
            return;
        }
        kv.tokens.extend_from_slice(&prompt[at..keep]);
    }
    slot.write(
        kv.ctx.ctx(),
        &kv.tokens,
        class,
        compiled.map(|(shape, _)| shape),
    );

    // Put the tail back: the turn still has to generate from the whole prompt.
    match decode_checkpointed(kv, prompt, keep) {
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
    draft: Option<&Arc<LlamaModel>>,
    snapshot: &mut Option<super::prompt_snapshot::SnapshotSlot>,
    model_path: &std::path::Path,
    model: &Arc<LlamaModel>,
    backend: &LlamaCppBackend,
    settings: &ModelSettings,
    prompt: &[LlamaToken],
    effective_ctx: usize,
    one_shot: bool,
    helper: bool,
    prime_waiting: bool,
    suspended: &mut Option<SuspendedKv>,
    compile_prefill: Option<&str>,
    prefill_retention_secs: Option<u64>,
    rebuild_prefill: bool,
) -> Result<PrefilledPrompt, ProviderError> {
    let n_ctx = u32::try_from(effective_ctx).map_err(|_| {
        ProviderError::ExecutionError(format!("Context size {effective_ctx} exceeds u32 range"))
    })?;

    if helper && prime_waiting {
        if suspended.is_none() {
            *suspended = suspend_for_helper(session);
        }
    } else if !helper && !one_shot {
        resume_suspended(
            session, suspended, draft, model, backend, settings, prompt, n_ctx,
        );
    } else if suspended.take().is_some() {
        tracing::info!(
            target: "giap::kv",
            event = "discarded",
            reason = "no waiting turn is left to continue it",
            "suspended chat cache deleted"
        );
    }

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
    // Set when a compile found this shape's snapshot missing or bad: the cold path below writes it.
    let mut rebuild_preamble: Option<usize> = None;
    if let Some(shape) = compile_prefill {
        // Judged on the cold path a real turn would take, never on a chat's resident cache.
        *session = None;
        if let Some(slot) = snapshot.as_mut() {
            if let Some(secs) = prefill_retention_secs {
                slot.retire_idle_helpers(secs);
            }
            let preamble = slot.compile_preamble(shape, prompt);
            let health = match preamble {
                None => super::prompt_snapshot::PreambleHealth::Unobserved,
                Some(preamble) => slot.preamble_health_at(prompt, preamble, rebuild_prefill),
            };
            if matches!(
                health,
                super::prompt_snapshot::PreambleHealth::Missing
                    | super::prompt_snapshot::PreambleHealth::Stale { .. }
                    | super::prompt_snapshot::PreambleHealth::Corrupt
            ) {
                rebuild_preamble = preamble;
            }
            tracing::info!(
                target: "giap::kv",
                event = "compiled",
                shape,
                verdict = ?health,
                preamble_tokens = preamble.unwrap_or(0),
                prompt_tokens = prompt.len(),
                "prefill compile checked this shape's snapshot; a bad one is rebuilt below"
            );
        }
    }

    let class = if helper {
        super::prompt_snapshot::SnapshotClass::Helper
    } else {
        super::prompt_snapshot::SnapshotClass::Shared
    };
    let plan = prefill_plan(
        session.as_ref().map(|kv| {
            // A text-only prompt carries no media, so it can never reproduce a
            // retained media head and nothing behind that head is resumable.
            let mut retained = if kv.head.is_some() {
                RetainedPrefix::opaque(kv)
            } else {
                RetainedPrefix::resumable(kv)
            };
            retained.worth_protecting &=
                retained_cache_protected(helper, !one_shot, kv.last_used.elapsed());
            retained
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

    // Nothing usable in memory: this is the cold start the snapshot exists for,
    // whether or not a context of the right size is already allocated.
    if matches!(
        plan,
        PrefillPlan::CreateContext | PrefillPlan::FullPrefillInPlace
    ) {
        if let Some(prefilled) = restore_preamble(
            session,
            snapshot,
            draft,
            model,
            backend,
            settings,
            prompt,
            n_ctx,
            effective_ctx,
        ) {
            return Ok(prefilled);
        }
    }

    match plan {
        PrefillPlan::ReusePrefix(resume) => {
            let kv = session
                .as_mut()
                .expect("ReusePrefix is only produced when a session is retained");

            match reuse_prefix(kv, prompt, resume) {
                Ok(resumed) => {
                    // Warm by definition: this turn reused a prefix.
                    write_snapshot_if_due(session, snapshot, prompt, false, class, None);
                    return Ok(PrefilledPrompt {
                        reused_prefix_tokens: resumed,
                        transient: None,
                        effective_ctx,
                    });
                }
                Err(PrefixReuseError::Refused) => {
                    tracing::debug!(
                        resume,
                        "the cache cannot resume there; prefilling the whole prompt"
                    );
                    // A resume the cache could not serve is a cold start too.
                    if let Some(prefilled) = restore_preamble(
                        session,
                        snapshot,
                        draft,
                        model,
                        backend,
                        settings,
                        prompt,
                        n_ctx,
                        effective_ctx,
                    ) {
                        return Ok(prefilled);
                    }
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
            match sacrificial_prefill(model, draft, backend, settings, prompt, transient_n_ctx) {
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

    // A compile rebuilding a preamble decodes exactly that far, saves it, then decodes the rest:
    // once the whole prompt is in, a windowed cache keeps no checkpoint that far back.
    if let (Some(keep), Some(slot)) = (rebuild_preamble, snapshot.as_mut()) {
        if keep < prompt.len() {
            full_prefill(
                session,
                draft,
                model,
                backend,
                settings,
                &prompt[..keep],
                n_ctx,
            )?;
            let kv = session
                .as_mut()
                .expect("full_prefill leaves a session on success");
            slot.write(kv.ctx.ctx(), &kv.tokens, class, compile_prefill);
            if let Err(e) = decode_checkpointed(kv, prompt, keep) {
                *session = None;
                return Err(e);
            }
            kv.tokens.extend_from_slice(&prompt[keep..]);
            return Ok(PrefilledPrompt {
                reused_prefix_tokens: 0,
                transient: None,
                effective_ctx,
            });
        }
    }

    let retained = session.is_some();
    match full_prefill(session, draft, model, backend, settings, prompt, n_ctx) {
        Ok(()) => {}
        Err(e) if retained => {
            // `full_prefill` dropped the poisoned context, so this attempt runs
            // against a freshly created one.
            tracing::warn!(error = %e, "prompt prefill failed; retrying with a new context");
            full_prefill(session, draft, model, backend, settings, prompt, n_ctx)?;
        }
        Err(e) => return Err(e),
    }
    // Cold: this path decoded the whole prompt.
    write_snapshot_if_due(
        session,
        snapshot,
        prompt,
        true,
        class,
        compile_prefill.zip(rebuild_preamble),
    );
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
    draft: Option<&Arc<LlamaModel>>,
    backend: &LlamaCppBackend,
    settings: &ModelSettings,
    prompt: &[LlamaToken],
    n_ctx: u32,
) -> Result<SessionKv, ProviderError> {
    let mut kv = SessionKv::create(model, draft, backend, n_ctx, settings)?;
    decode_tokens(kv.ctx.ctx_mut(), prompt, 0)?;
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
        Ok(resumed) => Some(resumed),
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
    memory_fit: Option<MemoryFit>,
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
            MtmdBitmap::from_buffer(mtmd_ctx, &img.bytes, false)
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
        memory_fit,
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
    // KV caches are never resident together. No drafter: a media prompt gives it
    // no token sequence to draft from.
    *session = None;
    let mut mm_settings = settings.clone();
    if mm_settings.n_ubatch.map_or(false, |u| u < 512) {
        mm_settings.n_ubatch = Some(512);
    }
    let mut kv = SessionKv::create(model, None, backend, n_ctx, &mm_settings)?;

    let n_batch = kv.ctx.ctx_mut().n_batch() as i32;
    let n_past = chunks
        .eval_chunks(mtmd_ctx, kv.ctx.ctx(), 0, 0, n_batch, true)
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
    let cap = context_cap(
        ctx.settings,
        ctx.context_limit,
        n_ctx_train,
        memory_max_ctx,
        ctx.memory_fit,
    );
    let token_budget = cap.saturating_sub(min_generation_headroom);
    let t_ctx_done = std::time::Instant::now();
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

    let t_render_done = std::time::Instant::now();
    let _ = ctx.log.write(
        &serde_json::json!({"applied_prompt": &template_result.prompt}),
        None,
    );
    let t_log_done = std::time::Instant::now();
    // (tokenize, validate, plan+decode) for the text path; the multimodal path
    // has no flat token vector to time the same way.
    let mut text_breakdown: Option<(u128, u128, u128)> = None;

    let (prompt_token_count, mut prefilled, prompt_tokens) = if !ctx.images.is_empty() {
        // The multimodal path prefills through mtmd chunks and never produces a
        // flat token vector, so a speculative draft has nothing to stand on --
        // which is consistent with the vision request bypassing KV retention too.
        let (ptc, prefilled) = prefill_multimodal(
            ctx.session,
            ctx.model,
            ctx.mtmd_ctx,
            ctx.backend,
            &template_result.prompt,
            ctx.images,
            ctx.context_limit,
            ctx.settings,
            ctx.memory_fit,
        )?;
        (ptc, prefilled, Vec::new())
    } else {
        let tokens = ctx
            .model
            .str_to_token(&template_result.prompt, AddBos::Never)
            .map_err(|e| ProviderError::ExecutionError(e.to_string()))?;
        let t_tok_done = std::time::Instant::now();
        let (ptc, ectx) = validate_and_compute_context(
            ctx.model,
            ctx.mtmd_ctx.is_some(),
            ctx.backend,
            tokens.len(),
            ctx.context_limit,
            ctx.settings,
            retained_kv_tokens,
            ctx.memory_fit,
        )?;
        let t_validate_done = std::time::Instant::now();
        let prefilled = prefill_prompt(
            ctx.session,
            ctx.draft,
            ctx.snapshot,
            ctx.model_path,
            ctx.model,
            ctx.backend,
            ctx.settings,
            &tokens,
            ectx,
            ctx.one_shot,
            ctx.helper,
            ctx.prime_waiting,
            ctx.suspended,
            ctx.compile_prefill.as_deref(),
            ctx.prefill_retention_secs,
            ctx.rebuild_prefill,
        )?;
        text_breakdown = Some((
            t_tok_done.duration_since(t_log_done).as_millis(),
            t_validate_done.duration_since(t_tok_done).as_millis(),
            t_validate_done.elapsed().as_millis(),
        ));
        (ptc, prefilled, tokens)
    };

    // Wait for the prefill to actually finish before timing it. The GPU
    // backends return from `decode` once the graph is submitted and bill the
    // remaining work to whoever reads the logits next -- the first `sample`,
    // which lands in time-to-first-token instead of here. Measured on Metal
    // (E2B, release): a 160-token warm prefill read 13-33 ms while 250-360 ms of
    // its work showed up as a gap between prefill end and first token. TTFT is
    // measured at the first emitted piece and was already honest; this makes
    // `prefill_ms`, and every rate derived from it, honest too.
    match prefilled.transient.as_mut() {
        Some(kv) => kv.session_ctx_mut().ctx_mut().synchronize(),
        None => {
            if let Some(kv) = ctx.session.as_mut() {
                kv.session_ctx_mut().ctx_mut().synchronize();
                kv.one_shot = ctx.one_shot || ctx.helper;
                kv.last_used = std::time::Instant::now();
                // A rewind into this prompt's reply resumes at its end.
                let tokens = std::mem::take(&mut kv.tokens);
                kv.checkpoint_window(&tokens);
                kv.tokens = tokens;
            }
        }
    }

    if let Some((tokenize_ms, validate_ms, plan_and_decode_ms)) = text_breakdown {
        tracing::debug!(
            target: "giap::kv",
            memory_estimate_ms = t_ctx_done.duration_since(prefill_started).as_millis(),
            render_ms = t_render_done.duration_since(t_ctx_done).as_millis(),
            log_write_ms = t_log_done.duration_since(t_render_done).as_millis(),
            tokenize_ms,
            validate_ms,
            plan_and_decode_ms,
            total_ms = prefill_started.elapsed().as_millis(),
            prompt_tokens = prompt_token_count,
            reused_prefix_tokens = prefilled.reused_prefix_tokens,
            "prefill breakdown"
        );
    }

    Ok(PreparedGeneration {
        template_result,
        prompt_tokens,
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
// `prompt_tokens` and `prompt_token_count` look redundant and are not: the
// count includes media chunks, while the token vector is text only, which is
// what the drafter has to be seeded with.
#[allow(clippy::too_many_arguments)]
pub(super) fn generation_loop(
    model: &LlamaModel,
    ctx: &mut SessionCtx,
    settings: &crate::local_model_registry::ModelSettings,
    prompt_tokens: &[LlamaToken],
    prompt_token_count: usize,
    effective_ctx: usize,
    decoded: &mut Vec<LlamaToken>,
    mut on_piece: impl FnMut(&str) -> Result<TokenAction, ProviderError>,
) -> Result<i32, ProviderError> {
    let mut sampler = build_sampler(model, settings);
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

    // Whole token sequence the target stands on: the drafter is given this on
    // every draft call, exactly as llama.cpp's server passes its slot's tokens.
    let mut sequence: Vec<LlamaToken> = prompt_tokens.to_vec();

    // Point the drafter at the prompt HERE rather than in each prefill helper.
    // The ledger is updated at six places across CreateContext, ReusePrefix,
    // SacrificialContext and the snapshot restore, and a `begin` missing from
    // any one of them does not error -- it leaves the drafter proposing against
    // a prompt the target has moved past, so every draft is rejected and the
    // speedup silently becomes a slowdown. This is the single point every path
    // passes through with the prefill finished.
    // A media prompt's tokens are not all text, so a session a text turn left
    // with a drafter does not speculate on it.
    let speculative = prompt_tokens.len() == prompt_token_count;
    if let Some(mtp) = ctx.mtp_mut().filter(|_| speculative) {
        mtp.begin(prompt_tokens)?;
    }

    // `pending` is always UNDECODED on entry to an iteration; whichever path
    // runs is responsible for decoding it.
    let mut pending = Some(sampler.sample(ctx.ctx_mut(), -1));
    sampler.accept(pending.expect("just sampled"));

    while (output_token_count as usize) < max_output {
        let Some(token) = pending else {
            exhausted_loop = false;
            break;
        };
        if model.is_eog_token(token) {
            exhausted_loop = false;
            break;
        }

        let mut emit = |tok: LlamaToken,
                        decoder: &mut encoding_rs::Decoder,
                        count: &mut i32|
         -> Result<bool, ProviderError> {
            *count += 1;
            let piece = model
                .token_to_piece(tok, decoder, true, None)
                .map_err(|e| {
                    ProviderError::ExecutionError(format!("Failed to decode token: {}", e))
                })?;
            Ok(!piece.is_empty() && matches!(on_piece(&piece)?, TokenAction::Stop))
        };

        if emit(token, &mut decoder, &mut output_token_count)? {
            exhausted_loop = false;
            break;
        }
        decoded.push(token);
        sequence.push(token);

        if speculative && ctx.mtp_mut().is_some() {
            let n_past = (sequence.len() - 1) as i32;
            let step = {
                let mtp = ctx.mtp_mut().expect("checked above");
                super::mtp::speculate(mtp, model, &mut sampler, n_past, token, &sequence)?
            };
            let mut stopped = false;
            for tok in step.tokens {
                if (output_token_count as usize) >= max_output {
                    stopped = true;
                    break;
                }
                if emit(tok, &mut decoder, &mut output_token_count)? {
                    exhausted_loop = false;
                    stopped = true;
                    break;
                }
                decoded.push(tok);
                sequence.push(tok);
            }
            if stopped {
                break;
            }
            pending = step.next;
            continue;
        }

        let next_tokens = [token];
        let mut next_batch = LlamaBatch::get_one(&next_tokens)
            .map_err(|e| ProviderError::ExecutionError(format!("Failed to create batch: {}", e)))?;
        ctx.ctx_mut()
            .decode(&mut next_batch)
            .map_err(|e| ProviderError::ExecutionError(format!("Decode failed: {}", e)))?;
        let t = sampler.sample(ctx.ctx_mut(), -1);
        sampler.accept(t);
        pending = Some(t);
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

/// Prefill a session directly, for tests that need a context standing on a
/// known prompt without the snapshot and plan machinery around it.
#[cfg(test)]
pub(super) fn decode_tokens_for_test(
    kv: &mut SessionKv,
    prompt: &[LlamaToken],
) -> Result<(), ProviderError> {
    decode_tokens(kv.ctx.ctx_mut(), prompt, 0)?;
    kv.tokens.clear();
    kv.tokens.extend_from_slice(prompt);
    Ok(())
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use crate::local_model_registry::ModelSettings;

    fn default_settings() -> ModelSettings {
        ModelSettings::default()
    }

    #[test]
    fn test_effective_context_size_uses_full_limit() {
        assert_eq!(
            effective_context_size(100, &default_settings(), 4096, 4096, None, None),
            4096
        );
    }

    #[test]
    fn test_effective_context_size_capped_by_limit() {
        assert_eq!(
            effective_context_size(100, &default_settings(), 1024, 8192, None, None),
            1024
        );
    }

    /// Without a limit from the registry or the caller, the memory estimate caps the window.
    #[test]
    fn test_effective_context_size_capped_by_memory() {
        assert_eq!(
            effective_context_size(100, &default_settings(), 0, 4096, Some(800), None),
            800
        );
    }

    #[test]
    fn test_effective_context_size_memory_smaller_than_prompt() {
        assert_eq!(
            effective_context_size(600, &default_settings(), 0, 4096, Some(700), None),
            700
        );
    }

    /// A limit the host set wins over the memory estimate: the host owns the budget.
    #[test]
    fn test_effective_context_size_a_set_limit_outranks_the_memory_estimate() {
        assert_eq!(
            effective_context_size(100, &default_settings(), 4096, 4096, Some(800), None),
            4096
        );
    }

    #[test]
    fn test_effective_context_size_zero_limit_uses_train() {
        assert_eq!(
            effective_context_size(100, &default_settings(), 0, 2048, None, None),
            2048
        );
    }

    #[test]
    fn test_effective_context_size_prompt_exceeds_all_limits() {
        assert_eq!(
            effective_context_size(5000, &default_settings(), 4096, 4096, None, None),
            4096
        );
    }

    #[test]
    fn test_context_cap_with_settings_override() {
        let mut settings = default_settings();
        settings.context_size = Some(2048);
        assert_eq!(context_cap(&settings, 4096, 8192, Some(1024), None), 2048);
    }

    #[test]
    fn test_context_cap_without_override() {
        assert_eq!(
            context_cap(&default_settings(), 4096, 8192, None, None),
            4096
        );
    }

    #[test]
    fn test_context_cap_memory_limited() {
        assert_eq!(
            context_cap(&default_settings(), 0, 8192, Some(2048), None),
            2048
        );
        assert_eq!(
            context_cap(&default_settings(), 4096, 8192, Some(2048), None),
            4096,
            "a limit the host set wins over the memory estimate"
        );
    }

    /// The Jetson's pin (16,384, stamped from a static budget) is narrowed when the memory free
    /// after the load cannot hold it, and kept exactly when it can; a smaller host cap is kept.
    #[test]
    fn test_context_cap_narrows_a_pin_to_the_memory_free_at_load() {
        use super::super::memory::{COMPUTE_RESERVE_BYTES, SAFETY_MARGIN_BYTES};
        let mut pinned = default_settings();
        pinned.context_size = Some(16_384);
        let per_token = 96 * 1024;
        let reserves = COMPUTE_RESERVE_BYTES + SAFETY_MARGIN_BYTES;
        let roomy = MemoryFit {
            kv_bytes_per_token: per_token,
            free_bytes: 16_384 * per_token + reserves,
        };
        let tight = MemoryFit {
            free_bytes: 8_192 * per_token + reserves,
            ..roomy
        };
        assert_eq!(context_cap(&pinned, 0, 32_768, None, Some(roomy)), 16_384);
        assert_eq!(context_cap(&pinned, 0, 32_768, None, Some(tight)), 8_192);
        assert_eq!(
            context_cap(&pinned, 0, 32_768, None, None),
            16_384,
            "off unified memory"
        );
        pinned.context_size = Some(6_144);
        assert_eq!(context_cap(&pinned, 0, 32_768, None, Some(tight)), 6_144);
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
            worth_protecting: true,
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
            worth_protecting: true,
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

    /// A windowed cache keeps only the last few hundred positions in its
    /// sliding-window layers, so a resume further back must come from a window
    /// checkpoint, and must then answer like a cache decoded from scratch.
    /// Refusing is the only other right answer. Gemma 4 E2B, the Orin's batch
    /// sizes; `#[ignore]` because it needs the GGUF.
    #[test]
    #[ignore = "requires a real GGUF on disk; run with --ignored"]
    fn a_resume_behind_the_window_answers_like_a_full_prefill() {
        use llama_cpp_2::model::params::LlamaModelParams;

        let Some(src) = live_model_path() else {
            eprintln!("skipping: no local GGUF found");
            return;
        };
        let tmp = tempfile::tempdir().expect("tempdir");
        let model_path = tmp.path().join("model.gguf");
        std::fs::hard_link(&src, &model_path).expect("hard link the model, never symlink it");
        let backend = shared_backend();
        let model = Arc::new(
            LlamaModel::load_from_file(
                backend.llama_backend(),
                &model_path,
                &LlamaModelParams::default(),
            )
            .expect("load model"),
        );
        assert!(
            sliding_window(&model) > 0,
            "the test needs a sliding-window model"
        );
        let settings = ModelSettings {
            n_batch: Some(512),
            n_ubatch: Some(256),
            ..Default::default()
        };
        let n_ctx: u32 = 4096;
        let text = |s: &str, bos: AddBos| model.str_to_token(s, bos).expect("tokenize");
        let shared = text(
            &"The pond keeps its own notes about the household. ".repeat(160),
            AddBos::Always,
        );
        let then_a = text(
            &" Kitchen sensors saw the fridge open at seven. ".repeat(110),
            AddBos::Never,
        );
        let then_b = text(" In one word, the weather today is", AddBos::Never);
        let first: Vec<LlamaToken> = shared.iter().chain(&then_a).copied().collect();
        let second: Vec<LlamaToken> = shared.iter().chain(&then_b).copied().collect();
        let last = second.len() - 1;

        // The cache as a turn leaves it: the first prompt, checkpointed.
        let mut kv = SessionKv::create(&model, None, backend, n_ctx, &settings).expect("ctx");
        decode_checkpointed(&mut kv, &first, 0).expect("prefill the first prompt");
        kv.tokens.extend_from_slice(&first);
        kv.checkpoint_window(&first);
        let pos_min = kv.ctx.ctx().kv_cache_seq_pos_min(0);
        assert!(
            pos_min as usize > shared.len().saturating_sub(kv.n_swa),
            "the window no longer covers the divergence ({pos_min}), so this must need a checkpoint"
        );

        let Ok(resumed) = reuse_prefix(&mut kv, &second[..last], shared.len()) else {
            panic!("the resume was refused although a checkpoint covers it");
        };
        assert!(
            resumed < shared.len(),
            "resumed at a checkpoint before the divergence"
        );
        decode_tokens(kv.ctx.ctx_mut(), &second[last..], last).expect("decode final token");
        let windowed = kv.ctx.ctx_mut().get_logits_ith(0).to_vec();
        eprintln!(
            "diverged at {}, resumed from a checkpoint at {resumed}, cache end was {}",
            shared.len(),
            first.len()
        );

        let fresh = {
            let mut kv = SessionKv::create(&model, None, backend, n_ctx, &settings).expect("ctx");
            decode_tokens(kv.ctx.ctx_mut(), &second[..last], 0).expect("full prefill");
            decode_tokens(kv.ctx.ctx_mut(), &second[last..], last).expect("decode final token");
            kv.ctx.ctx_mut().get_logits_ith(0).to_vec()
        };
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
            top5(&windowed),
            top5(&fresh),
            "a resume behind the window predicts differently from a full prefill"
        );

        // Without the checkpoint the same resume must be refused, never served.
        let mut bare = SessionKv::create(&model, None, backend, n_ctx, &settings).expect("ctx");
        decode_checkpointed(&mut bare, &first, 0).expect("prefill the first prompt");
        bare.tokens.extend_from_slice(&first);
        bare.checkpoints.clear();
        assert!(matches!(
            reuse_prefix(&mut bare, &second[..last], shared.len()),
            Err(PrefixReuseError::Refused)
        ));
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
        use crate::llamacpp::prompt_snapshot::{SnapshotClass, SnapshotSlot};
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
            let mut kv = SessionKv::create(&model, None, backend, n_ctx, &settings).expect("ctx");
            decode_tokens(kv.ctx.ctx_mut(), &prefix, 0).expect("prefill preamble");
            kv.tokens.extend_from_slice(&prefix);
            let t = std::time::Instant::now();
            assert!(
                slot.write(kv.ctx.ctx(), &kv.tokens, SnapshotClass::Shared, None),
                "snapshot must be written"
            );
            eprintln!(
                "SAVE  {} tokens in {} ms",
                kv.tokens.len(),
                t.elapsed().as_millis()
            );
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
            let mut kv = SessionKv::create(&model, None, backend, n_ctx, &settings).expect("ctx");
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
            decode_tokens(kv.ctx.ctx_mut(), &prompt[n..last], n).expect("decode tail");
            decode_tokens(kv.ctx.ctx_mut(), &prompt[last..], last).expect("decode final token");
            eprintln!(
                "restore + tail decode: {} ms",
                started.elapsed().as_millis()
            );
            kv.ctx.ctx_mut().get_logits_ith(0).to_vec()
        };

        // 3. The control: the same prompt, decoded from nothing.
        let fresh = {
            let started = std::time::Instant::now();
            let mut kv = SessionKv::create(&model, None, backend, n_ctx, &settings).expect("ctx");
            let last = prompt.len() - 1;
            decode_tokens(kv.ctx.ctx_mut(), &prompt[..last], 0).expect("full prefill");
            decode_tokens(kv.ctx.ctx_mut(), &prompt[last..], last).expect("decode final token");
            eprintln!(
                "full prefill:          {} ms",
                started.elapsed().as_millis()
            );
            kv.ctx.ctx_mut().get_logits_ith(0).to_vec()
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
            let mut kv = SessionKv::create(&model, None, backend, n_ctx, &settings).expect("ctx");
            let last = prompt.len() - 1;
            decode_tokens(kv.ctx.ctx_mut(), &prefix, 0).expect("prefix");
            decode_tokens(kv.ctx.ctx_mut(), &prompt[prefix.len()..last], prefix.len())
                .expect("tail");
            decode_tokens(kv.ctx.ctx_mut(), &prompt[last..], last).expect("final");
            kv.ctx.ctx_mut().get_logits_ith(0).to_vec()
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
            let mut kv = SessionKv::create(&model, None, backend, n_ctx, &settings).expect("ctx");
            let too_small = prefix.len() - 1;
            assert!(
                kv.context_mut()
                    .state_seq_load_file(&snapshot_path, 0, too_small)
                    .is_err(),
                "a capacity below the stored token count must fail -- if this starts passing, \
                 max_tokens has stopped being a buffer size and the reasoning above is stale"
            );

            let mut kv = SessionKv::create(&model, None, backend, n_ctx, &settings).expect("ctx");
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
        let mut kv = SessionKv::create(&model, None, backend, n_ctx, &settings).expect("ctx");
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
                None,
                &mut snapshot,
                &model_path,
                &model,
                backend,
                &settings,
                prompt,
                n_ctx,
                false,
                false,
                false,
                &mut None,
                None,
                None,
                false,
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

    /// A real model loaded the way the Orin's registry loads E4B, its files hard-linked into a
    /// temp root that also holds every cache the engine writes. `GIAP_TEST_GGUF`,
    /// `GIAP_TEST_DRAFT` and `GIAP_TEST_NCTX` override it.
    struct LiveEngine {
        model: Arc<LlamaModel>,
        draft: Option<Arc<LlamaModel>>,
        settings: ModelSettings,
        model_path: std::path::PathBuf,
        n_ctx: u32,
        _env: env_lock::EnvGuard<'static>,
        tmp: tempfile::TempDir,
    }

    fn live_engine(default_n_ctx: u32) -> Option<LiveEngine> {
        use llama_cpp_2::model::params::LlamaModelParams;

        let env_path = |name: &str| std::env::var_os(name).map(std::path::PathBuf::from);
        let src = env_path("GIAP_TEST_GGUF").or_else(live_model_path)?;
        let n_ctx = std::env::var("GIAP_TEST_NCTX")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default_n_ctx);
        let tmp = tempfile::tempdir().expect("tempdir; set TMPDIR to the model's filesystem");
        let link = |src: &std::path::Path, name: &str| {
            let dst = tmp.path().join(name);
            let real = std::fs::canonicalize(src).expect("resolve the model path");
            std::fs::hard_link(&real, &dst).expect("hard link the model, never symlink it");
            dst
        };
        let model_path = link(&src, "model.gguf");
        let draft_path = env_path("GIAP_TEST_DRAFT").map(|d| link(&d, "draft.gguf"));
        let env = env_lock::lock_env([("GOOSE_PATH_ROOT", Some(tmp.path().to_str().unwrap()))]);

        let backend = shared_backend();
        let params = LlamaModelParams::default().with_n_gpu_layers(99);
        let load = |path: &std::path::Path| {
            Arc::new(
                LlamaModel::load_from_file(backend.llama_backend(), path, &params)
                    .expect("load model"),
            )
        };
        let model = load(&model_path);
        let draft = draft_path.as_deref().map(load);
        let settings = ModelSettings {
            context_size: Some(n_ctx),
            n_batch: Some(512),
            n_ubatch: Some(512),
            type_k: Some("f16".to_string()),
            type_v: Some("f16".to_string()),
            n_gpu_layers: Some(99),
            flash_attention: Some(true),
            n_threads: Some(4),
            ..Default::default()
        };
        Some(LiveEngine {
            model,
            draft,
            settings,
            model_path,
            n_ctx,
            _env: env,
            tmp,
        })
    }

    /// Exactly `n` tokens of `sentence` repeated, so prompt lengths are what a test says.
    fn filler(
        model: &LlamaModel,
        sentence: &str,
        n: usize,
        bos: llama_cpp_2::model::AddBos,
    ) -> Vec<LlamaToken> {
        let mut tokens = model
            .str_to_token(&sentence.repeat(n / 4 + 8), bos)
            .expect("tokenize");
        assert!(tokens.len() >= n, "filler too short for {n} tokens");
        tokens.truncate(n);
        tokens
    }

    /// The suspend file exists only while a helper holds the context for a waiting turn; the
    /// turn's next prompt resumes from it, and a helper nobody waits on never writes one.
    #[test]
    #[ignore = "requires a real GGUF on disk; run with --ignored"]
    fn a_waiting_turn_resumes_from_its_suspend_file() {
        use llama_cpp_2::model::AddBos;

        let Some(live) = live_engine(8192) else {
            eprintln!("skipping: no local GGUF found");
            return;
        };
        let backend = shared_backend();
        let prime = filler(
            &live.model,
            "The pond keeps its own notes about the household and answers when asked. ",
            2500,
            AddBos::Always,
        );
        let helper = filler(
            &live.model,
            "You are the pond's schedule helper: you read calendars and report back. ",
            1200,
            AddBos::Always,
        );
        let tail = filler(
            &live.model,
            " And then what happened next? ",
            64,
            AddBos::Never,
        );
        let continued: Vec<LlamaToken> = prime.iter().chain(&tail).copied().collect();

        let mut session = None;
        let mut snapshot = None;
        let mut suspended = None;
        macro_rules! prefill {
            ($prompt:expr, $helper:expr, $prime_waiting:expr) => {
                prefill_prompt(
                    &mut session,
                    live.draft.as_ref(),
                    &mut snapshot,
                    &live.model_path,
                    &live.model,
                    backend,
                    &live.settings,
                    $prompt,
                    live.n_ctx as usize,
                    false,
                    $helper,
                    $prime_waiting,
                    &mut suspended,
                    None,
                    None,
                    false,
                )
                .expect("prefill")
                .reused_prefix_tokens
            };
        }

        prefill!(&prime, false, false);
        prefill!(&helper, true, true);
        let path = suspended
            .as_ref()
            .expect("a waiting turn's cache must be set aside, not evicted")
            .path
            .clone();
        assert!(
            path.exists(),
            "the suspend file must be on disk while the helper runs"
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode =
                |p: &std::path::Path| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
            assert_eq!(
                mode(&path),
                0o600,
                "the file holds conversation: owner only"
            );
            assert_eq!(mode(path.parent().unwrap()), 0o700);
        }

        let t = std::time::Instant::now();
        let reused = prefill!(&continued, false, false);
        eprintln!(
            "resumed a {}-token turn in {} ms",
            prime.len(),
            t.elapsed().as_millis()
        );
        assert!(
            suspended.is_none() && !path.exists(),
            "a resumed turn leaves no file behind"
        );
        assert_eq!(
            reused,
            prime.len(),
            "the continuing turn must reuse its whole suspended prefix, not rebuild it"
        );

        prefill!(&helper, true, false);
        assert!(
            suspended.is_none(),
            "a helper nobody waits on must never put a conversation on disk"
        );
    }

    /// Two compile calls of one shape build its missing snapshot, leave a good one untouched and
    /// rebuild one whose bytes changed; a second shape opening the same way keeps its own beside it,
    /// and a real turn of either restores its whole preamble.
    #[test]
    #[ignore = "requires a real GGUF on disk; run with --ignored"]
    fn a_compile_builds_keeps_and_repairs_a_shapes_snapshot() {
        use llama_cpp_2::model::AddBos;

        let Some(live) = live_engine(8192) else {
            eprintln!("skipping: no local GGUF found");
            return;
        };
        let backend = shared_backend();
        let preamble = filler(
            &live.model,
            "The pond keeps its own notes about the household and answers when asked. ",
            1500,
            AddBos::Always,
        );
        let with = |tail: &str| -> Vec<LlamaToken> {
            let tail = filler(&live.model, tail, 40, AddBos::Never);
            preamble.iter().chain(&tail).copied().collect()
        };
        let (first, second, real) = (
            with("Alpha is the first of two compile calls. "),
            with("Begin the second compile call here. "),
            with("Today is Tuesday and the person asks about dinner. "),
        );
        assert_ne!(
            first[preamble.len()],
            second[preamble.len()],
            "tails must differ at once"
        );

        let mut session = None;
        let mut snapshot = None;
        let mut suspended = None;
        macro_rules! prefill {
            ($prompt:expr, $compile:expr) => {
                prefill_prompt(
                    &mut session,
                    live.draft.as_ref(),
                    &mut snapshot,
                    &live.model_path,
                    &live.model,
                    backend,
                    &live.settings,
                    $prompt,
                    live.n_ctx as usize,
                    false,
                    false,
                    false,
                    &mut suspended,
                    $compile,
                    None,
                    false,
                )
                .expect("prefill")
                .reused_prefix_tokens
            };
        }
        let files = |snapshot: &Option<crate::llamacpp::prompt_snapshot::SnapshotSlot>| {
            snapshot
                .as_ref()
                .map_or_else(Vec::new, |s| s.snapshot_paths())
        };

        prefill!(&first, Some("test-shape"));
        prefill!(&second, Some("test-shape"));
        let built = files(&snapshot);
        assert_eq!(
            built.len(),
            1,
            "a compile of a shape with no snapshot must build one"
        );
        let path = built[0].clone();
        let written_at = std::fs::metadata(&path).unwrap().modified().unwrap();

        prefill!(&first, Some("test-shape"));
        prefill!(&second, Some("test-shape"));
        assert_eq!(
            std::fs::metadata(&path).unwrap().modified().unwrap(),
            written_at,
            "a valid snapshot must be kept, not rewritten"
        );

        let mut bytes = std::fs::read(&path).unwrap();
        let middle = bytes.len() / 2;
        bytes[middle] ^= 0xff;
        std::fs::write(&path, &bytes).unwrap();
        prefill!(&first, Some("test-shape"));
        prefill!(&second, Some("test-shape"));
        let repaired = files(&snapshot);
        assert_eq!(repaired.len(), 1, "a corrupt snapshot must be rebuilt");
        assert_ne!(
            std::fs::read(&repaired[0]).unwrap(),
            bytes,
            "the rebuilt snapshot must not be the corrupted bytes"
        );

        let other: Vec<LlamaToken> = preamble[..600]
            .iter()
            .copied()
            .chain(filler(
                &live.model,
                "A helper reads one step of a task and writes what it found. ",
                1000,
                AddBos::Never,
            ))
            .collect();
        let other_with = |tail: &str| -> Vec<LlamaToken> {
            let tail = filler(&live.model, tail, 40, AddBos::Never);
            other.iter().chain(&tail).copied().collect()
        };
        prefill!(
            &other_with("Alpha is the first of two compile calls. "),
            Some("other-shape")
        );
        prefill!(
            &other_with("Begin the second compile call here. "),
            Some("other-shape")
        );
        assert_eq!(
            files(&snapshot).len(),
            2,
            "a shape opening as another does must keep its own snapshot beside it"
        );

        session = None;
        let reused = prefill!(&real, None);
        assert_eq!(
            reused,
            preamble.len(),
            "a real turn of the shape must restore exactly the compiled preamble"
        );
        session = None;
        let reused = prefill!(
            &other_with("A step asks what the household eats on Tuesdays. "),
            None
        );
        assert_eq!(
            reused,
            other.len(),
            "a real turn of the second shape must restore its own preamble"
        );
    }

    /// A helper turn beside the prime (a throwaway context) against handing it the one context and
    /// rebuilding the prime from its preamble snapshot or a suspend file. Stop the pond first.
    #[test]
    #[ignore = "requires a real GGUF on disk; run with --ignored"]
    fn what_a_helper_turn_costs_the_prime() {
        use crate::llamacpp::prompt_snapshot::{SnapshotClass, SnapshotSlot};
        use llama_cpp_2::model::AddBos;
        use std::time::Instant;

        let env_num = |name: &str, default: usize| {
            std::env::var(name)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(default)
        };
        let Some(live) = live_engine(16384) else {
            eprintln!("skipping: no local GGUF found");
            return;
        };
        let (model, draft, settings, model_path, tmp) = (
            &live.model,
            &live.draft,
            &live.settings,
            &live.model_path,
            &live.tmp,
        );
        let n_ctx = live.n_ctx;
        let backend = shared_backend();
        let preamble_len = env_num("GIAP_TEST_PREAMBLE", 3000);
        let role_len = env_num("GIAP_TEST_ROLE", 1100);
        let task_len = 300;
        let text = |sentence: &str, n: usize, bos: AddBos| filler(model, sentence, n, bos);
        let preamble = text(
            "The pond keeps its own notes about the household and answers when asked. ",
            preamble_len,
            AddBos::Always,
        );
        let role = text(
            "You are the pond's schedule helper: you read calendars and report back. ",
            role_len,
            AddBos::Always,
        );
        let task = text(
            "Find a free hour on Saturday for the two of them and say why it fits. ",
            task_len,
            AddBos::Never,
        );
        let helper: Vec<LlamaToken> = role.iter().chain(&task).copied().collect();
        let probe = model
            .str_to_token(" So the answer is", AddBos::Never)
            .expect("tokenize probe")[0];

        let mem_available_mib = || -> Option<i64> {
            let info = std::fs::read_to_string("/proc/meminfo").ok()?;
            let kb = info.lines().find_map(|l| l.strip_prefix("MemAvailable:"))?;
            kb.trim()
                .trim_end_matches("kB")
                .trim()
                .parse::<i64>()
                .ok()
                .map(|kb| kb / 1024)
        };
        let taken = |before: Option<i64>| {
            before
                .zip(mem_available_mib())
                .map_or("n/a".to_string(), |(b, a)| format!("{} MiB", b - a))
        };
        let ms = |t: Instant| t.elapsed().as_millis();
        let answer_after_probe = |kv: &mut SessionKv, pos: usize| {
            decode_tokens(kv.ctx.ctx_mut(), &[probe], pos).expect("decode probe");
            kv.ctx
                .ctx_mut()
                .get_logits_ith(0)
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i)
                .expect("logits")
        };

        let mut slot =
            SnapshotSlot::for_model(&model_path, n_ctx, &settings).expect("slot for a real model");
        for (name, tokens, class) in [
            ("prime preamble", &preamble, SnapshotClass::Shared),
            ("role preamble", &role, SnapshotClass::Helper),
        ] {
            let mut kv =
                SessionKv::create(&model, draft.as_ref(), backend, n_ctx, &settings).expect("ctx");
            decode_tokens(kv.ctx.ctx_mut(), tokens, 0).expect("prefill preamble");
            let t = Instant::now();
            assert!(
                slot.write(kv.ctx.ctx(), tokens, class, None),
                "{name} snapshot must be written"
            );
            eprintln!("{name}: {} tokens saved in {} ms", tokens.len(), ms(t));
        }

        let mut disagreements = Vec::new();
        for history_len in [2048usize, 4096, 8192] {
            let history = text(
                "Earlier the family talked about the garden, the school run and dinner. ",
                history_len,
                AddBos::Never,
            );
            let prime: Vec<LlamaToken> = preamble.iter().chain(&history).copied().collect();
            if prime.len() + 1 > n_ctx as usize {
                eprintln!("history {history_len}: does not fit n_ctx {n_ctx}, skipped");
                continue;
            }
            let suspend_path = tmp.path().join(format!("prime-{history_len}.kv"));

            let mut session: Option<SessionKv> = None;
            let before = mem_available_mib();
            let t = Instant::now();
            full_prefill(
                &mut session,
                draft.as_ref(),
                &model,
                backend,
                &settings,
                &prime,
                n_ctx,
            )
            .expect("cold prime");
            let cold_ms = ms(t);
            let prime_mem = taken(before);

            let t = Instant::now();
            let suspend_bytes = session
                .as_ref()
                .expect("prime")
                .ctx
                .ctx()
                .state_seq_save_file(&suspend_path, 0, &prime)
                .expect("suspend the prime");
            let suspend_ms = ms(t);
            let cold_answer = answer_after_probe(session.as_mut().expect("prime"), prime.len());

            let before = mem_available_mib();
            let t = Instant::now();
            let side_n_ctx =
                sacrificial_context_size(helper.len(), settings.max_output_tokens, n_ctx);
            let side = sacrificial_prefill(
                &model,
                draft.as_ref(),
                backend,
                &settings,
                &helper,
                side_n_ctx,
            );
            let side_ms = ms(t);
            let side_mem = taken(before);
            let side_ok = side.is_ok();
            drop(side);

            let helper_in = |session: &mut Option<SessionKv>, slot: &mut SnapshotSlot| {
                let t = Instant::now();
                let restored = seed_from_snapshot(
                    session,
                    draft.as_ref(),
                    &model,
                    backend,
                    &settings,
                    slot,
                    &helper,
                    n_ctx,
                )
                .expect("helper seed");
                assert_eq!(
                    restored,
                    Some(role.len()),
                    "the role snapshot must seed the helper"
                );
                ms(t)
            };

            let helper_seed_ms = helper_in(&mut session, &mut slot);
            let t = Instant::now();
            session = None;
            let mut kv =
                SessionKv::create(&model, draft.as_ref(), backend, n_ctx, &settings).expect("ctx");
            let (tokens, _) = kv
                .context_mut()
                .state_seq_load_file(&suspend_path, 0, n_ctx as usize)
                .expect("resume the suspended prime");
            let resume_ms = ms(t);
            assert_eq!(
                tokens.len(),
                prime.len(),
                "the suspend file must hold the whole prime"
            );
            let resumed_answer = answer_after_probe(&mut kv, prime.len());
            session = Some(kv);

            helper_in(&mut session, &mut slot);
            let t = Instant::now();
            let restored = seed_from_snapshot(
                &mut session,
                draft.as_ref(),
                &model,
                backend,
                &settings,
                &mut slot,
                &prime,
                n_ctx,
            )
            .expect("rebuild the prime");
            let rebuild_ms = ms(t);
            assert_eq!(
                restored,
                Some(preamble.len()),
                "the preamble snapshot must seed the prime"
            );
            let rebuilt_answer = answer_after_probe(session.as_mut().expect("prime"), prime.len());
            drop(session);
            let _ = std::fs::remove_file(&suspend_path);

            eprintln!(
                "history {history_len} (prime {} tok): cold {cold_ms} ms, context {prime_mem} | \
                 side context ({side_n_ctx}): {} {side_ms} ms, {side_mem} | helper seeded {helper_seed_ms} ms | \
                 prime rebuilt {rebuild_ms} ms | suspend {suspend_ms} ms, {:.0} MiB, resume {resume_ms} ms",
                prime.len(),
                if side_ok { "ok" } else { "FAILED" },
                suspend_bytes as f64 / 1_048_576.0,
            );
            if rebuilt_answer != cold_answer || resumed_answer != cold_answer {
                disagreements.push((history_len, cold_answer, rebuilt_answer, resumed_answer));
            }
        }
        assert!(
            disagreements.is_empty(),
            "a rebuilt or resumed prime predicted a different next token than the cold one \
             (history, cold, rebuilt, resumed): {disagreements:?}"
        );
    }

    /// The process's one llama.cpp backend.
    ///
    /// `LlamaCppBackend::new()` treats a second initialisation as `unreachable!`
    /// — correctly, since the runtime holds the only one for the life of the
    /// process — so two live tests each building their own panics the second.
    pub(crate) fn shared_backend() -> &'static LlamaCppBackend {
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
            worth_protecting: true,
        };
        let prompt = tokens(0..2000);
        assert_eq!(
            prefill_plan(Some(opaque), &prompt, 4096),
            PrefillPlan::FullPrefillInPlace
        );
    }

    #[test]
    fn a_cache_one_shot_work_filled_protects_nothing() {
        // A compaction summary stopped halfway leaves its prompt in the cache.
        // The next chat turn takes the slot instead of running in a throwaway
        // context beside it, as it would beside a chat's cache.
        let leftover = tokens(10_000..23_000);
        let chat_turn = tokens(0..4_000);
        let one_shot = RetainedPrefix {
            worth_protecting: false,
            ..retained(&leftover, 16_384)
        };
        assert_eq!(
            prefill_plan(Some(one_shot), &chat_turn, 16_384),
            PrefillPlan::FullPrefillInPlace
        );
        assert_eq!(
            prefill_plan(Some(retained(&leftover, 16_384)), &chat_turn, 16_384),
            PrefillPlan::SacrificialContext,
            "a chat's cache of that size is still protected"
        );
    }

    #[test]
    fn a_helper_is_never_kept_out_of_the_one_context() {
        let now = std::time::Duration::ZERO;
        assert!(!retained_cache_protected(true, true, now));
        assert!(
            retained_cache_protected(false, true, now),
            "a chat turn must still be shielded, or the helper case proves nothing"
        );
    }

    #[test]
    fn a_new_conversation_takes_over_a_chat_cache_left_idle() {
        let second = std::time::Duration::from_secs(1);
        assert!(chat_cache_protected(
            true,
            CHAT_CACHE_PROTECTED_FOR - second
        ));
        assert!(!chat_cache_protected(true, CHAT_CACHE_PROTECTED_FOR));
        assert!(
            chat_cache_protected(false, CHAT_CACHE_PROTECTED_FOR * 60),
            "a side call never evicts a chat's cache, however long it has sat"
        );

        // The Orin's case: a 3,456-token voice turn against a 6,939-token cache.
        let earlier_chat = tokens(10_000..16_939);
        let new_conversation = tokens(0..3_456);
        let idle = RetainedPrefix {
            worth_protecting: chat_cache_protected(true, CHAT_CACHE_PROTECTED_FOR),
            ..retained(&earlier_chat, 16_384)
        };
        assert_eq!(
            prefill_plan(Some(idle), &new_conversation, 16_384),
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
            worth_protecting: true,
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
