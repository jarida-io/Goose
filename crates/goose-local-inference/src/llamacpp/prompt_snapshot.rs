//! The shared preamble's KV cache, kept on disk across processes and models.
//!
//! # Why
//!
//! The retained [`SessionKv`](super::inference_engine::SessionKv) makes the
//! SECOND turn of a session cheap and does nothing at all for the first. Every
//! cold start — a new session, a model swap, a restart — re-decodes a preamble
//! that is byte-identical to the one decoded a minute earlier, because the
//! system prompt and the tool schemas do not change between turns.
//!
//! Measured on the host this was written against: a 6,685-token preamble
//! prefills in **24.1 s** at 277 tok/s, and there were 26–40 new sessions in a
//! day. The same KV state is ~366 MiB on disk for a 56 KiB/token model, which
//! reads back in a fraction of a second. Recomputing it is between one and two
//! orders of magnitude more expensive than reading it.
//!
//! # What is stored, and what is deliberately not
//!
//! Only the **stable prefix**: the longest run of tokens every prompt for this
//! model has begun with since the process started. That is the system prompt
//! and the tool schemas, and it is impersonal by construction — GIAP injects
//! memories and turn context inside the user message, which diverges and is
//! therefore excluded by the very definition of the prefix.
//!
//! This is a privacy property, not an optimisation: a KV blob is a numeric
//! encoding of the tokens that produced it, so a snapshot of a whole prompt
//! would be a household's conversation sitting in a cache file. Truncating to
//! the common prefix means the file cannot contain anything a second prompt did
//! not also contain.
//!
//! # Why it can never be shared between models
//!
//! KV state is the model's own attention cache: its layer count, head
//! dimensions and rope parameters are baked into the bytes. A snapshot is valid
//! for exactly one `(model file, context size, layout)` triple, which is what
//! [`SnapshotKey`] names. Different models keep different files and evict
//! nothing of each other's — which is the point, because a model swap is the
//! most expensive cold start there is.
//!
//! # Failure is always "prefill normally"
//!
//! Every path here returns `Option`/`bool` rather than an error. The engine's
//! standing invariant is that a bad cache never fails a turn, and a snapshot is
//! strictly an optimisation: a missing, stale, corrupt or mismatched file must
//! be indistinguishable from having no file at all.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::token::LlamaToken;

use super::inference_engine::common_prefix_len;

/// Bumped when anything changes that alters the meaning of the stored bytes:
/// the llama.cpp state format, the sequence id written, or the key's own
/// composition.
///
/// It is part of the file name rather than a header inside the file, so an
/// incompatible snapshot is not *read and rejected* — it is never opened, and
/// the old one is left for a cleanup pass rather than being parsed by code that
/// may not understand it.
const SNAPSHOT_FORMAT_VERSION: u32 = 1;

/// The only sequence a retained context uses. `reuse_prefix` and
/// `full_prefill` both address sequence 0 explicitly; a snapshot that wrote or
/// read any other sequence would restore into a cache nothing else looks at.
const SNAPSHOT_SEQ_ID: i32 = 0;

/// Below this a snapshot is not worth its own file.
///
/// Deliberately far above `REUSE_MIN_TOKENS` (256), which governs an in-memory
/// resume that costs nothing but a `seq_rm`. A snapshot costs a write of tens
/// to hundreds of megabytes and a read on every cold start, so it has to save
/// meaningfully more than that overhead. At the measured 277 tok/s, 1,024
/// tokens is ~3.7 s of prefill — comfortably worth a disk read, while a
/// several-hundred-token preamble is not.
const SNAPSHOT_MIN_TOKENS: usize = 1024;

/// The most one snapshot may hold.
///
/// Writing measured 67 MiB/s, so the file size IS the write latency: at
/// 56 KiB/token a 16,384-token prefix would be 900 MiB and take a quarter of a
/// minute to save, on a turn. Past a few thousand tokens the prefill being saved
/// is already amortised across every future cold start, and the marginal token
/// buys less than it costs to store. The cap is on tokens rather than bytes
/// because the byte cost is only known after the write.
const SNAPSHOT_MAX_TOKENS: usize = 6144;

/// Token depths the prefix ladder hashes at.
///
/// # Why a ladder and not one hash
///
/// A pond does not have ONE prompt shape. Chat, memory extraction, session
/// titling and the proactive reviewer each build their own system prompt, and
/// they share nothing at the start — `EXTRACTION_PROMPT` is its own ~450-token
/// text. A single stable prefix per model is the intersection across all of
/// them, which collapses toward nothing, and then over-fits to whichever shape
/// happened to run first. On a live pond that produced a 5,341-token snapshot
/// every later caller rejected.
///
/// So a model keeps SEVERAL snapshots and each prompt gets the one that matches
/// it deepest. The ladder is how that choice is made without reading anything:
/// each stored snapshot records the hash of its own first 256, 512, 1024 …
/// tokens, an incoming prompt computes the same, and the deepest rung both agree
/// on is how much they provably share. Reading a 300 MiB file to discover it was
/// the wrong one is exactly the cost this avoids.
///
/// The rungs are hashes, so nothing about a prompt's contents is recoverable
/// from the index — which matters, because the index is written to disk beside
/// caches whose whole privacy argument is that they hold only the shared
/// preamble.
const LADDER_DEPTHS: &[usize] = &[256, 512, 1024, 2048, 4096, 8192, 16384];

/// How many snapshots one `(model, n_ctx, layout)` may keep.
///
/// One per prompt shape is the point, and a pond has a handful: chat, extraction,
/// titling, the reviewer, the summariser. Six leaves room for a new caller
/// without letting a runaway fill the disk — at 18 KiB/token a 4,000-token
/// snapshot is ~70 MiB, so six is the difference between a cache and a problem.
const MAX_SNAPSHOTS: usize = 6;

/// How stale a `last_used` stamp may get before the index is rewritten.
const TOUCH_WRITE_INTERVAL_SECS: u64 = 3600;

/// Hash of `tokens[..depth]` for every depth the tokens actually reach.
///
/// Returns `(depth, hash)` pairs, shallowest first. A prompt shorter than the
/// first rung yields nothing and is simply not a snapshot candidate — it could
/// not clear `SNAPSHOT_MIN_TOKENS` either.
fn ladder(tokens: &[LlamaToken]) -> Vec<(usize, u64)> {
    let mut out = Vec::new();
    let mut h = Fnv::new();
    let mut consumed = 0usize;
    for &depth in LADDER_DEPTHS {
        if tokens.len() < depth {
            break;
        }
        // Folded incrementally: each rung continues the previous one's state, so
        // the ladder costs one pass over the tokens rather than one per rung.
        for t in &tokens[consumed..depth] {
            h.write(&t.0.to_le_bytes());
        }
        consumed = depth;
        out.push((depth, h.finish()));
    }
    out
}

/// How deeply two ladders agree, in tokens. Zero means they do not share even
/// the shallowest rung, which is the ordinary case for two different callers.
fn ladder_agreement(a: &[(usize, u64)], b: &[(usize, u64)]) -> usize {
    a.iter()
        .zip(b)
        .take_while(|((da, ha), (db, hb))| da == db && ha == hb)
        .map(|((d, _), _)| *d)
        .last()
        .unwrap_or(0)
}

/// One stored snapshot, as the index remembers it.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct IndexEntry {
    file: String,
    tokens: usize,
    ladder: Vec<(usize, u64)>,
    /// Seconds since the epoch, for eviction. Not correctness — a wrong clock
    /// costs a suboptimal eviction and nothing else.
    last_used: u64,
}

/// The set of snapshots for one model, and which one a prompt should use.
///
/// Small enough to read and rewrite whole on every change; it holds a handful of
/// entries and is read once per model load.
#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
struct SnapshotIndex {
    entries: Vec<IndexEntry>,
}

impl SnapshotIndex {
    /// The entry sharing the deepest ladder with `want`, if any shares at all.
    ///
    /// Ties go to the entry with MORE tokens: two snapshots agreeing to the same
    /// rung are equally right about the part that was compared, and the longer
    /// one saves more of the decode. The prefix check at load time is still what
    /// makes the choice safe — the ladder narrows, it does not prove.
    fn best_match(&self, want: &[(usize, u64)], prompt_tokens: usize) -> Option<&IndexEntry> {
        self.entries
            .iter()
            .filter(|e| {
                // Cheap refusals, in the order that rejects most for least. Each
                // one saves a full read of a file that CANNOT serve this prompt,
                // and a read is tens to hundreds of megabytes -- the single
                // largest waste this type can commit.
                //
                // 1. A snapshot at least as long as the prompt leaves nothing to
                //    decode, which `load` rejects anyway.
                if e.tokens >= prompt_tokens {
                    return false;
                }
                // 2. The ladder has to agree at least as deep as the snapshot's
                //    own deepest rung. Agreeing less means the stored prefix
                //    diverges from this prompt BEFORE it ends, so it is not a
                //    prefix of it and no amount of reading will change that.
                let own_depth = e.ladder.last().map_or(0, |(d, _)| *d);
                ladder_agreement(&e.ladder, want) >= own_depth && own_depth > 0
            })
            .max_by_key(|e| e.tokens)
    }
}

/// Identity of a snapshot: everything that must match for the stored bytes to
/// mean what they meant when they were written.
///
/// The model is identified by path, length and modification time rather than by
/// a content hash. Hashing several gigabytes on every model load would cost
/// more than the snapshot saves, and the failure this needs to catch is a
/// re-downloaded or swapped file — which changes length or mtime. A file
/// replaced by a byte-identical copy with a preserved mtime is
/// indistinguishable, and correctly so: the bytes are the same.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct SnapshotKey {
    model_path: PathBuf,
    model_len: u64,
    model_mtime_secs: i64,
    n_ctx: u32,
    /// Context-construction settings that change the cache's shape. Anything
    /// `build_context_params` reads and that alters KV layout belongs here.
    layout: u64,
}

impl SnapshotKey {
    pub(super) fn new(model_path: &Path, n_ctx: u32, layout: u64) -> Option<Self> {
        let meta = std::fs::metadata(model_path).ok()?;
        let mtime = meta
            .modified()
            .ok()?
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            // A model older than the epoch is not a real case; treating it as
            // an unknown time rather than refusing keeps the key total.
            .unwrap_or(-1);
        Some(Self {
            model_path: model_path.to_path_buf(),
            model_len: meta.len(),
            model_mtime_secs: mtime,
            n_ctx,
            layout,
        })
    }

    /// A file name that changes whenever any component of the key changes.
    ///
    /// Hashed with FNV-1a rather than `DefaultHasher`, and that is the whole
    /// reason this function exists instead of a derive: `DefaultHasher`'s output
    /// is explicitly not stable across Rust releases, so a toolchain upgrade
    /// would silently orphan every snapshot on disk and quietly re-introduce the
    /// cold start this module exists to remove. Nothing here is
    /// security-sensitive — a collision costs a rejected load, because the
    /// loaded tokens are still checked against the prompt.
    pub(super) fn family(&self) -> u64 {
        let mut h = Fnv::new();
        h.write(self.model_path.to_string_lossy().as_bytes());
        h.write(&self.model_len.to_le_bytes());
        h.write(&self.model_mtime_secs.to_le_bytes());
        h.write(&self.n_ctx.to_le_bytes());
        h.write(&self.layout.to_le_bytes());
        h.finish()
    }

    /// The index listing every snapshot for this model.
    ///
    /// The version is in the NAME, so an index an older build cannot understand
    /// is never opened rather than opened and misread.
    pub(super) fn index_name(&self) -> String {
        format!(
            "v{SNAPSHOT_FORMAT_VERSION}-{:016x}.index.json",
            self.family()
        )
    }
}

/// FNV-1a, 64-bit. Small, dependency-free and — unlike `DefaultHasher` —
/// specified, so the same key yields the same file name on every host and every
/// toolchain.
struct Fnv(u64);

impl Fnv {
    fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }
    fn write(&mut self, bytes: &[u8]) {
        for b in bytes {
            self.0 ^= u64::from(*b);
            self.0 = self.0.wrapping_mul(0x1000_0000_01b3);
        }
        // Length-mixed so that concatenated fields cannot alias: without this,
        // ("ab", "c") and ("a", "bc") hash identically and two different models
        // could share a file.
        self.0 ^= bytes.len() as u64;
        self.0 = self.0.wrapping_mul(0x1000_0000_01b3);
    }
    fn finish(&self) -> u64 {
        self.0
    }
}

/// Where snapshots live, under the goose data directory.
///
/// Its own directory rather than beside the weights: these are derived files
/// that may be deleted at any time, and a stray `.kv` next to a GGUF invites
/// exactly the kind of "is this part of the model?" question that gets a model
/// deleted by a cleanup script.
const SNAPSHOT_DIR: &str = "prompt-cache";

/// Where the snapshot's lifecycle is narrated.
///
/// An explicit target, and that is load-bearing rather than tidy. GIAP's
/// `tracing_setup.rs` carves this whole crate to ERROR
/// (`goose_local_inference=error`) because goose narrates every turn, so an
/// `info!` raised from here under the default target is invisible on exactly the
/// pond it was written for. `EnvFilter` matches the TARGET, not the module, so
/// naming one escapes the carve — the same trick GIAP's own `giap::trace` events
/// use. Kept distinct from `giap::trace` so the whole life of a cache greps out
/// on its own.
const KV_TARGET: &str = "giap::kv";

/// Settings that could change what a stored cache means.
///
/// Conservative on purpose: `n_batch` almost certainly does not alter the stored
/// state, but including it costs one bit of key space and excluding it wrongly
/// costs a corrupt restore. `n_threads` is excluded because it cannot affect
/// anything but scheduling.
fn layout_fingerprint(settings: &crate::local_model_registry::ModelSettings) -> u64 {
    let flash = match settings.flash_attention {
        Some(true) => 1u64,
        Some(false) => 2,
        None => 0,
    };
    let batch = u64::from(settings.n_batch.unwrap_or(0));
    flash | (batch << 8)
}

/// Per-loaded-model snapshot state.
///
/// Lives beside the retained session on [`LoadedModel`](super::inference_engine::LoadedModel)
/// and is dropped with it, so a slot eviction discards the in-memory half while
/// leaving the file for the next load to find.
pub(super) struct SnapshotSlot {
    dir: PathBuf,
    key: SnapshotKey,
    /// What is on disk for this model, and how to choose between them.
    index: SnapshotIndex,
    /// One tracked prefix per prompt SHAPE, keyed by the shallowest ladder rung.
    ///
    /// Keyed rather than single because a pond runs several callers against one
    /// model and they share nothing at the start. Folding them into one prefix
    /// is what produced an intersection of nothing and, before that, a snapshot
    /// over-fitted to whichever caller ran first.
    buckets: HashMap<u64, Bucket>,
}

/// One prompt shape's converging prefix.
#[derive(Default)]
struct Bucket {
    stable: Option<Vec<LlamaToken>>,
    /// Whether folding a prompt in has ever SHORTENED the prefix. A prefix
    /// derived from one prompt IS that prompt; only a shrink proves two
    /// genuinely different prompts were seen.
    intersected: bool,
    /// Written or found this process, so the expensive write happens at most
    /// once per shape per slot.
    settled: bool,
}

impl SnapshotSlot {
    /// Build a slot for one model at one context size, or `None` when the model
    /// file cannot be read -- in which case snapshots are simply off for this
    /// request, which is the same as every other failure here.
    pub(super) fn for_model(
        model_path: &Path,
        n_ctx: u32,
        settings: &crate::local_model_registry::ModelSettings,
    ) -> Option<Self> {
        let layout = layout_fingerprint(settings);
        let key = SnapshotKey::new(model_path, n_ctx, layout)?;
        Some(Self::new(
            crate::paths::Paths::in_data_dir(SNAPSHOT_DIR),
            key,
        ))
    }

    /// Whether this slot already describes `(model_path, n_ctx)`.
    pub(super) fn matches(&self, model_path: &Path, n_ctx: u32) -> bool {
        self.key.n_ctx == n_ctx && self.key.model_path == model_path
    }

    pub(super) fn new(dir: PathBuf, key: SnapshotKey) -> Self {
        let index_path = dir.join(key.index_name());
        let index = std::fs::read_to_string(&index_path)
            .ok()
            .and_then(|raw| serde_json::from_str::<SnapshotIndex>(&raw).ok())
            .unwrap_or_default();

        tracing::info!(
            target: KV_TARGET,
            event = "created",
            model = %key.model_path.display(),
            n_ctx = key.n_ctx,
            known_snapshots = index.entries.len(),
            dir = %dir.display(),
            "kv snapshot slot created"
        );
        Self {
            dir,
            key,
            index,
            buckets: HashMap::new(),
        }
    }

    /// The converged prefix length for a prompt's shape, for tests.
    #[cfg(test)]
    fn stable_len(&self, prompt: &[LlamaToken]) -> Option<usize> {
        let bucket = Self::bucket_of(prompt)?;
        self.buckets
            .get(&bucket)
            .and_then(|b| b.stable.as_ref())
            .map(Vec::len)
    }

    /// Every snapshot file this model currently has, newest write last.
    ///
    /// Test-only: production chooses one via the ladder rather than enumerating.
    #[cfg(test)]
    pub(super) fn snapshot_paths(&self) -> Vec<PathBuf> {
        self.index
            .entries
            .iter()
            .map(|e| self.dir.join(&e.file))
            .collect()
    }

    fn index_path(&self) -> PathBuf {
        self.dir.join(self.key.index_name())
    }

    fn save_index(&self) {
        let Ok(raw) = serde_json::to_string(&self.index) else {
            return;
        };
        let _ = std::fs::create_dir_all(&self.dir);
        if let Err(e) = std::fs::write(self.index_path(), raw) {
            tracing::debug!(target: KV_TARGET, error = %e, "kv snapshot index not written");
        }
    }

    /// The bucket a prompt belongs to: its shallowest ladder rung.
    ///
    /// `None` for a prompt too short to reach the first rung, which could never
    /// clear `SNAPSHOT_MIN_TOKENS` either.
    fn bucket_of(prompt: &[LlamaToken]) -> Option<u64> {
        ladder(prompt).first().map(|(_, h)| *h)
    }

    /// Fold one prompt into ITS OWN shape's stable prefix.
    pub(super) fn observe(&mut self, prompt: &[LlamaToken]) {
        let Some(bucket) = Self::bucket_of(prompt) else {
            return;
        };
        let entry = self.buckets.entry(bucket).or_default();
        match &mut entry.stable {
            None => entry.stable = Some(prompt.to_vec()),
            Some(stable) => {
                let shared = common_prefix_len(stable, prompt);
                if shared < stable.len() {
                    entry.intersected = true;
                }
                stable.truncate(shared);
            }
        }
    }

    /// How much of THIS prompt's shape is worth persisting, if a write is due.
    pub(super) fn writable_prefix(&self, prompt: &[LlamaToken]) -> Option<usize> {
        let bucket = Self::bucket_of(prompt)?;
        let entry = self.buckets.get(&bucket)?;
        if entry.settled || !entry.intersected {
            return None;
        }
        let len = entry
            .stable
            .as_ref()
            .map_or(0, Vec::len)
            .min(SNAPSHOT_MAX_TOKENS);
        (len >= SNAPSHOT_MIN_TOKENS).then_some(len)
    }

    /// Persist the first `tokens.len()` positions of `ctx`'s sequence 0.
    ///
    /// The caller must have already trimmed the cache to exactly that length --
    /// llama.cpp writes whatever the sequence holds, so trimming is how the
    /// "only the stable prefix" property is enforced rather than merely
    /// intended.
    pub(super) fn write(&mut self, ctx: &LlamaContext<'_>, tokens: &[LlamaToken]) -> bool {
        let Some(bucket) = Self::bucket_of(tokens) else {
            return false;
        };
        let started = std::time::Instant::now();
        if let Err(e) = std::fs::create_dir_all(&self.dir) {
            tracing::warn!(
                target: KV_TARGET,
                event = "discarded",
                reason = "cache directory could not be created",
                error = %e,
                "kv snapshot discarded"
            );
            self.buckets.entry(bucket).or_default().settled = true;
            return false;
        }

        let rungs = ladder(tokens);
        let file = format!(
            "v{SNAPSHOT_FORMAT_VERSION}-{:016x}-{:016x}.kv",
            self.key.family(),
            bucket
        );
        let final_path = self.dir.join(&file);
        let tmp_path = final_path.with_extension("kv.part");

        match ctx.state_seq_save_file(&tmp_path, SNAPSHOT_SEQ_ID, tokens) {
            Ok(bytes) => {
                if let Err(e) = std::fs::rename(&tmp_path, &final_path) {
                    tracing::warn!(
                        target: KV_TARGET,
                        event = "discarded",
                        reason = "rename into place failed",
                        error = %e,
                        "kv snapshot discarded"
                    );
                    let _ = std::fs::remove_file(&tmp_path);
                    self.buckets.entry(bucket).or_default().settled = true;
                    return false;
                }
                self.index.entries.retain(|e| e.file != file);
                self.index.entries.push(IndexEntry {
                    file: file.clone(),
                    tokens: tokens.len(),
                    ladder: rungs,
                    last_used: now_secs(),
                });
                self.evict_beyond_cap();
                self.save_index();

                tracing::info!(
                    target: KV_TARGET,
                    event = "saved",
                    tokens = tokens.len(),
                    bytes,
                    kib_per_token = bytes as f64 / 1024.0 / tokens.len().max(1) as f64,
                    elapsed_ms = started.elapsed().as_millis() as u64,
                    snapshots_for_model = self.index.entries.len(),
                    path = %final_path.display(),
                    "kv snapshot saved; this prompt shape now resumes from disk"
                );
                self.buckets.entry(bucket).or_default().settled = true;
                true
            }
            Err(e) => {
                tracing::warn!(
                    target: KV_TARGET,
                    event = "discarded",
                    reason = "llama.cpp refused to write the sequence state",
                    error = %e,
                    "kv snapshot discarded"
                );
                let _ = std::fs::remove_file(&tmp_path);
                self.buckets.entry(bucket).or_default().settled = true;
                false
            }
        }
    }

    /// Restore the snapshot that matches `prompt` deepest, if one does.
    pub(super) fn load(
        &mut self,
        ctx: &mut LlamaContext<'_>,
        prompt: &[LlamaToken],
    ) -> Option<usize> {
        let rungs = ladder(prompt);
        let chosen = self.index.best_match(&rungs, prompt.len())?.clone();
        let path = self.dir.join(&chosen.file);
        if !path.exists() {
            self.forget(&chosen.file, "index named a file that is gone");
            return None;
        }
        let started = std::time::Instant::now();

        // The context's capacity, NOT the prompt length. `max_tokens` sizes the
        // output buffer llama.cpp writes the token list into, so passing the
        // prompt length makes any snapshot LONGER than the current prompt fail
        // outright with "token count in sequence state file exceeded capacity".
        let capacity = ctx.n_ctx() as usize;
        let (tokens, bytes) = match ctx.state_seq_load_file(&path, SNAPSHOT_SEQ_ID, capacity) {
            Ok(loaded) => loaded,
            Err(e) => {
                tracing::warn!(
                    target: KV_TARGET,
                    event = "discarded",
                    reason = "file present but llama.cpp could not read it",
                    error = %e,
                    path = %path.display(),
                    "kv snapshot discarded; prefilling normally"
                );
                self.retire(&chosen.file, "unreadable");
                clear_seq(ctx);
                return None;
            }
        };

        let shared = common_prefix_len(&tokens, prompt);
        if shared < tokens.len() || shared < SNAPSHOT_MIN_TOKENS || shared >= prompt.len() {
            tracing::info!(
                target: KV_TARGET,
                event = "rejected",
                reason = if shared < tokens.len() {
                    "the stored prefix is not a prefix of this prompt"
                } else if shared < SNAPSHOT_MIN_TOKENS {
                    "the shared run is too short to be worth reusing"
                } else {
                    "the stored prefix covers the whole prompt, leaving nothing to decode"
                },
                loaded_tokens = tokens.len(),
                shared,
                prompt_tokens = prompt.len(),
                "kv snapshot rejected; prefilling normally"
            );
            // Only retire on a genuine mismatch. A snapshot that merely COVERS
            // this prompt is right about a longer one and belongs to a shape
            // still in use, so deleting it would throw away a good cache because
            // one short prompt arrived.
            if shared < tokens.len() {
                self.retire(&chosen.file, "this pond's prompts do not begin with it");
            }
            clear_seq(ctx);
            return None;
        }

        self.touch(&chosen.file);
        tracing::info!(
            target: KV_TARGET,
            event = "loaded",
            reused_tokens = shared,
            prompt_tokens = prompt.len(),
            bytes,
            elapsed_ms = started.elapsed().as_millis() as u64,
            chose = %chosen.file,
            of_snapshots = self.index.entries.len(),
            "kv snapshot loaded; the preamble prefill is skipped"
        );
        Some(shared)
    }

    /// Record a use, and only pay a write when it would change the eviction
    /// order. `last_used` exists to pick a victim; rewriting the file on every
    /// successful load is a synchronous write per turn to move a timestamp
    /// nothing will read until the cap is exceeded.
    fn touch(&mut self, file: &str) {
        let now = now_secs();
        let stale = self
            .index
            .entries
            .iter()
            .find(|e| e.file == file)
            .is_some_and(|e| now.saturating_sub(e.last_used) > TOUCH_WRITE_INTERVAL_SECS);
        if let Some(e) = self.index.entries.iter_mut().find(|e| e.file == file) {
            e.last_used = now;
        }
        if stale {
            self.save_index();
        }
    }

    /// Forget an index entry without touching disk (the file is already gone).
    fn forget(&mut self, file: &str, reason: &str) {
        self.index.entries.retain(|e| e.file != file);
        self.save_index();
        tracing::debug!(target: KV_TARGET, reason, file, "kv snapshot forgotten");
    }

    /// Delete a snapshot that cannot serve this pond, and allow a replacement.
    fn retire(&mut self, file: &str, reason: &str) {
        let path = self.dir.join(file);
        let _ = std::fs::remove_file(&path);
        self.index.entries.retain(|e| e.file != file);
        self.save_index();
        tracing::info!(
            target: KV_TARGET,
            event = "discarded",
            reason = %reason,
            scope = "file deleted; a replacement may be written once a shared prefix is known",
            path = %path.display(),
            "kv snapshot retired"
        );
        // Un-settle every shape: whichever one owned this file may now write a
        // better version, and the cheap ones will simply find nothing to do.
        for b in self.buckets.values_mut() {
            b.settled = false;
        }
    }

    /// Keep the newest `MAX_SNAPSHOTS`, deleting the rest.
    fn evict_beyond_cap(&mut self) {
        while self.index.entries.len() > MAX_SNAPSHOTS {
            let Some((i, victim)) = self
                .index
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, e)| e.last_used)
                .map(|(i, e)| (i, e.file.clone()))
            else {
                break;
            };
            let _ = std::fs::remove_file(self.dir.join(&victim));
            self.index.entries.remove(i);
            tracing::info!(
                target: KV_TARGET,
                event = "discarded",
                reason = "least recently used, over the per-model cap",
                cap = MAX_SNAPSHOTS,
                file = %victim,
                "kv snapshot evicted"
            );
        }
    }
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

impl Drop for SnapshotSlot {
    /// The slot lives on the loaded model, so this fires when the model slot is
    /// evicted — the moment the in-memory half of the cache goes away and the
    /// next turn for this model becomes a cold start again. The FILE survives;
    /// that is the whole point, and saying which of the two was discarded is the
    /// difference between reading this log and guessing at it.
    fn drop(&mut self) {
        tracing::info!(
            target: KV_TARGET,
            event = "discarded",
            reason = "model slot evicted",
            scope = "in-memory slot only; the snapshot file is kept",
            model = %self.key.model_path.display(),
            n_ctx = self.key.n_ctx,
            "kv snapshot slot discarded"
        );
    }
}

/// Drop everything a failed restore may have left in the cache.
///
/// Not merely tidiness: `state_seq_load_file` can populate part of a sequence
/// before failing, and the caller's next move is a full prefill from position
/// zero, which assumes an empty cache.
fn clear_seq(ctx: &mut LlamaContext<'_>) {
    let _ = ctx.clear_kv_cache_seq(Some(SNAPSHOT_SEQ_ID as u32), None, None);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slot() -> SnapshotSlot {
        SnapshotSlot::new(
            PathBuf::from("/tmp/does-not-need-to-exist"),
            SnapshotKey {
                model_path: PathBuf::from("/m.gguf"),
                model_len: 1,
                model_mtime_secs: 2,
                n_ctx: 4096,
                layout: 0,
            },
        )
    }

    /// Tokens long enough to reach the first ladder rung, so they have a shape.
    /// `lead` decides which shape: it is what the bucket hash sees.
    fn shaped(lead: i32, len: usize) -> Vec<LlamaToken> {
        (0..len)
            .map(|i| LlamaToken(if i < 256 { lead } else { i as i32 }))
            .collect()
    }

    /// The stable prefix is an intersection, not a memory of the last prompt.
    #[test]
    fn the_stable_prefix_shrinks_to_what_every_prompt_shares() {
        let mut s = slot();
        let a = shaped(7, 2000);
        s.observe(&a);
        assert_eq!(s.stable_len(&a), Some(2000));

        let mut b = a.clone();
        b[1500] = LlamaToken(-99);
        s.observe(&b);
        assert_eq!(s.stable_len(&a), Some(1500));

        // Never grows back.
        s.observe(&b);
        assert_eq!(s.stable_len(&a), Some(1500));
    }

    /// THE POINT OF SEVERAL CACHES. Chat, memory extraction, titling and the
    /// reviewer each build their own system prompt and share nothing at the
    /// start. Folded into one prefix they intersect to nothing; kept apart, each
    /// keeps its own and every caller gets a warm start.
    #[test]
    fn two_prompt_shapes_keep_separate_prefixes_instead_of_cancelling_out() {
        let mut s = slot();

        let chat_a = shaped(1, 3000);
        let mut chat_b = chat_a.clone();
        chat_b[2500] = LlamaToken(-1);

        let sched_a = shaped(2, 3000);
        let mut sched_b = sched_a.clone();
        sched_b[2000] = LlamaToken(-2);

        // Interleaved, as a real pond runs them.
        s.observe(&chat_a);
        s.observe(&sched_a);
        s.observe(&chat_b);
        s.observe(&sched_b);

        assert_eq!(
            s.stable_len(&chat_a),
            Some(2500),
            "chat keeps its own prefix"
        );
        assert_eq!(
            s.stable_len(&sched_a),
            Some(2000),
            "the scheduler keeps its own, and did not shorten chat's"
        );
        assert!(s.writable_prefix(&chat_a).is_some());
        assert!(s.writable_prefix(&sched_a).is_some());
    }

    /// The ladder is what picks a cache without reading one. Deepest agreement
    /// wins; a shape that agrees about nothing is not a candidate at all.
    #[test]
    fn the_deepest_matching_ladder_wins_and_a_stranger_matches_nothing() {
        let deep = ladder(&shaped(1, 4000));
        let shallow_same_shape = ladder(&shaped(1, 600));
        let other_shape = ladder(&shaped(2, 4000));

        assert_eq!(ladder_agreement(&deep, &shallow_same_shape), 512);
        assert_eq!(
            ladder_agreement(&deep, &other_shape),
            0,
            "different openings must not share even the first rung"
        );

        let index = SnapshotIndex {
            entries: vec![
                IndexEntry {
                    file: "shallow.kv".into(),
                    tokens: 600,
                    ladder: shallow_same_shape,
                    last_used: 1,
                },
                IndexEntry {
                    file: "deep.kv".into(),
                    tokens: 4000,
                    ladder: deep.clone(),
                    last_used: 1,
                },
            ],
        };
        // A longer prompt of the same shape: the deeper snapshot serves it.
        let incoming = ladder(&shaped(1, 5000));
        assert_eq!(
            index.best_match(&incoming, 5000).map(|e| e.file.as_str()),
            Some("deep.kv")
        );
        assert_eq!(
            index
                .best_match(&other_shape, 5000)
                .map(|e| e.file.as_str()),
            None,
            "a prompt shape with no stored cache must get nothing rather than the wrong one"
        );

        // The refusals that save a read. Each would otherwise cost tens to
        // hundreds of megabytes to discover.
        assert_eq!(
            index.best_match(&incoming, 600).map(|e| e.file.as_str()),
            None,
            "a snapshot at least as long as the prompt leaves nothing to decode"
        );
        let diverges_early = {
            let mut p = shaped(1, 5000);
            p[700] = LlamaToken(-7);
            ladder(&p)
        };
        assert_eq!(
            index
                .best_match(&diverges_early, 5000)
                .map(|e| e.file.as_str()),
            Some("shallow.kv"),
            "the 4,000-token snapshot diverges from this prompt at token 700, so it cannot be \
             a prefix of it and must be refused FROM THE LADDER rather than by reading 4,000 \
             tokens of KV to find out -- while the 600-token one still genuinely is a prefix \
             and should be served"
        );
    }

    #[test]
    fn a_prefix_below_the_threshold_is_not_written() {
        // Both cases intersect first, so the threshold is what is under test
        // rather than the intersection requirement beside it.
        let mut s = slot();
        let a = shaped(5, SNAPSHOT_MIN_TOKENS - 1);
        let mut b = a.clone();
        b[300] = LlamaToken(-5);
        s.observe(&a);
        s.observe(&b);
        assert_eq!(s.writable_prefix(&a), None);

        let mut s = slot();
        let a = shaped(6, SNAPSHOT_MIN_TOKENS + 50);
        let mut b = a.clone();
        b[SNAPSHOT_MIN_TOKENS] = LlamaToken(-6);
        s.observe(&a);
        s.observe(&b);
        assert_eq!(s.writable_prefix(&a), Some(SNAPSHOT_MIN_TOKENS));
    }

    /// `load` must size the read buffer from the CONTEXT, never from the prompt.
    ///
    /// Asserted against the source because the two spellings are otherwise
    /// indistinguishable from outside: both end in `None` and both retire the
    /// file, so a test driving `load` cannot tell a capacity error from an
    /// honest "this is not a prefix". The difference is only visible in the log
    /// and in what the pond does next — and on a live pond the wrong one meant
    /// every cold start read a file, failed, and deleted nothing, forever.
    ///
    /// The companion assertion is in the live test, which pins llama.cpp's own
    /// semantics: a capacity below the stored token count really does fail.
    #[test]
    fn the_read_buffer_is_sized_from_the_context_not_the_prompt() {
        const SRC: &str = include_str!("prompt_snapshot.rs");
        // Split rather than slice: a byte range into source text can land
        // inside a UTF-8 sequence, and this file has plenty of prose that is
        // not ASCII.
        let body = SRC
            .split("fn load(")
            .nth(1)
            .expect("load is defined in this file")
            .split("\n    }")
            .next()
            .expect("split always yields one");

        assert!(
            body.contains("ctx.n_ctx() as usize"),
            "load no longer sizes its read buffer from the context"
        );
        assert!(
            !body.contains("SNAPSHOT_SEQ_ID, prompt.len()"),
            "load is passing the prompt length as the read capacity again -- that is not a \
             filter, it is a buffer size, and any snapshot longer than the prompt will fail \
             to read at all"
        );
    }

    /// A prefix derived from a single prompt IS that prompt, and writing it
    /// produced the defect this guard exists for: on a live pond the first write
    /// stored 5,341 tokens because every prompt in that process came from one
    /// conversation and shared its whole history. Every later session then
    /// rejected it, and the log filled with a failed load on each cold start.
    #[test]
    fn a_prefix_that_has_never_intersected_anything_is_not_written() {
        let long = shaped(9, SNAPSHOT_MIN_TOKENS + 200);

        // Three prompts, all identical: nothing has been learned. This is a
        // whole conversation's worth of turns.
        let mut s = slot();
        for _ in 0..3 {
            s.observe(&long);
        }
        assert_eq!(
            s.writable_prefix(&long),
            None,
            "agreeing prompts prove nothing about what a DIFFERENT session shares"
        );

        // One genuinely different prompt, and now the prefix is an intersection.
        let mut diverged = long.clone();
        let cut = SNAPSHOT_MIN_TOKENS + 100;
        diverged[cut] = LlamaToken(-1);
        s.observe(&diverged);
        assert_eq!(
            s.writable_prefix(&long),
            Some(cut),
            "a shrink is the evidence, and the shared run is what gets written"
        );
    }

    /// Writing is once per slot. Repeating it would put a multi-hundred-megabyte
    /// write in front of a turn, every turn, for a file that is already there.
    #[test]
    fn a_settled_slot_never_offers_another_write() {
        let mut s = slot();
        let a = shaped(8, SNAPSHOT_MIN_TOKENS + 50);
        let mut b = a.clone();
        b[SNAPSHOT_MIN_TOKENS] = LlamaToken(-8);
        s.observe(&a);
        s.observe(&b);
        assert!(s.writable_prefix(&a).is_some());
        // Set directly: this is the state `load` and `write` both leave behind,
        // and the assertion is about what `writable_prefix` does with it.
        let bucket = SnapshotSlot::bucket_of(&a).expect("a shaped prompt has a bucket");
        s.buckets.get_mut(&bucket).expect("observed").settled = true;
        assert_eq!(s.writable_prefix(&a), None);
    }

    /// Every component of the key has to reach the file name. A key that
    /// ignored `n_ctx` would hand a 16K snapshot to a 4K context, which is not
    /// a slow turn but a corrupt one.
    #[test]
    fn every_key_component_changes_the_file_name() {
        let base = SnapshotKey {
            model_path: PathBuf::from("/a.gguf"),
            model_len: 10,
            model_mtime_secs: 20,
            n_ctx: 4096,
            layout: 1,
        };
        let name = base.index_name();

        let variants = [
            SnapshotKey {
                model_path: PathBuf::from("/b.gguf"),
                ..base.clone()
            },
            SnapshotKey {
                model_len: 11,
                ..base.clone()
            },
            SnapshotKey {
                model_mtime_secs: 21,
                ..base.clone()
            },
            SnapshotKey {
                n_ctx: 8192,
                ..base.clone()
            },
            SnapshotKey {
                layout: 2,
                ..base.clone()
            },
        ];
        for v in &variants {
            assert_ne!(
                v.index_name(),
                name,
                "changing a key component must change the file name: {v:?}"
            );
        }
        // ...and the same key is the same name, or nothing is ever reused.
        assert_eq!(base.index_name(), name);
    }

    /// The hasher must distinguish where one field ended and the next began.
    ///
    /// Asserted against `Fnv` directly rather than through `SnapshotKey`, and
    /// that is the point: every field of the key but the path is fixed-width, so
    /// no pair of keys can alias today and a test written at that level passes
    /// whether the mixing is there or not. It was, and it proved nothing --
    /// deleting the mixing left it green. The property still has to hold,
    /// because the key gains a second variable-width field the moment anyone
    /// adds one, so it is pinned where it can actually fail.
    #[test]
    fn the_hasher_distinguishes_where_one_field_ends_and_the_next_begins() {
        let mut split = Fnv::new();
        split.write(b"ab");
        split.write(b"c");

        let mut other = Fnv::new();
        other.write(b"a");
        other.write(b"bc");

        assert_ne!(
            split.finish(),
            other.finish(),
            "two different field sequences hashed to one value; a key gaining a second \
             variable-width field would then let two models share a snapshot"
        );
    }

    /// A slot is rebuilt when the window moves. The sacrificial path runs in a
    /// smaller context, so a slot that answered `true` on `n_ctx` alone would
    /// hand a 16K snapshot to a 2K context -- not a slow turn, a corrupt one.
    #[test]
    fn a_slot_stops_matching_when_the_window_or_the_model_moves() {
        let s = slot();
        assert!(s.matches(Path::new("/m.gguf"), 4096));
        assert!(!s.matches(Path::new("/m.gguf"), 2048));
        assert!(!s.matches(Path::new("/other.gguf"), 4096));
    }

    /// Flash attention changes how attention is computed, so a cache written
    /// under one setting must not be handed to a context built under the other.
    /// `None` is its own value: "unset" is not the same claim as "off".
    #[test]
    fn the_layout_fingerprint_separates_settings_that_could_change_the_bytes() {
        use crate::local_model_registry::ModelSettings;

        let off = ModelSettings {
            flash_attention: Some(false),
            ..Default::default()
        };
        let on = ModelSettings {
            flash_attention: Some(true),
            ..Default::default()
        };
        let unset = ModelSettings::default();

        let f = layout_fingerprint;
        assert_ne!(f(&off), f(&on));
        assert_ne!(f(&unset), f(&off));
        assert_ne!(f(&unset), f(&on));

        let batched = ModelSettings {
            n_batch: Some(512),
            ..Default::default()
        };
        assert_ne!(f(&batched), f(&unset));
        // ...and identical settings must agree, or nothing is ever reused.
        assert_eq!(f(&ModelSettings::default()), f(&unset));
    }

    /// The name carries the format version, so an incompatible snapshot is never
    /// opened rather than opened and rejected.
    #[test]
    fn the_format_version_is_part_of_the_name() {
        let k = SnapshotKey {
            model_path: PathBuf::from("/a.gguf"),
            model_len: 1,
            model_mtime_secs: 1,
            n_ctx: 1,
            layout: 0,
        };
        assert!(k
            .index_name()
            .starts_with(&format!("v{SNAPSHOT_FORMAT_VERSION}-")));
        assert!(k.index_name().ends_with(".index.json"));
    }
}
