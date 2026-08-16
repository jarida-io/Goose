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
    pub(super) fn file_name(&self) -> String {
        let mut h = Fnv::new();
        h.write(self.model_path.to_string_lossy().as_bytes());
        h.write(&self.model_len.to_le_bytes());
        h.write(&self.model_mtime_secs.to_le_bytes());
        h.write(&self.n_ctx.to_le_bytes());
        h.write(&self.layout.to_le_bytes());
        // The version is in the NAME so an incompatible file is never opened.
        format!("v{SNAPSHOT_FORMAT_VERSION}-{:016x}.kv", h.finish())
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
    /// The longest prefix every prompt seen this process has shared.
    ///
    /// `None` until the first prompt. Monotonically shrinking afterwards, which
    /// is what makes it *stable* rather than merely *recent*: a prefix that has
    /// survived N prompts is a prefix the N+1th is likely to share too.
    stable: Option<Vec<LlamaToken>>,
    /// Set once a file has been written or found this process, so the expensive
    /// write happens at most once per slot.
    settled: bool,
    /// How many prompts have been folded in.
    ///
    /// A prefix derived from ONE prompt is that prompt, not a prefix. Writing it
    /// produced a 5,341-token snapshot on a live pond that every later session
    /// rejected, because consecutive prompts inside a single conversation share
    /// their whole history and the intersection had never actually intersected
    /// anything.
    observations: usize,
    /// Whether folding a prompt in has ever SHORTENED the prefix.
    ///
    /// The count alone is not evidence: three prompts from one session agree
    /// completely and still say nothing about what a different session shares.
    /// A shrink is proof that two genuinely different prompts have been seen.
    intersected: bool,
}

impl SnapshotSlot {
    /// Build a slot for one model at one context size, or `None` when the model
    /// file cannot be read -- in which case snapshots are simply off for this
    /// request, which is the same as every other failure here.
    ///
    /// `settings` contributes the layout fingerprint: anything
    /// `build_context_params` sets that could change what the stored bytes mean.
    /// `n_ctx` and `swa_full` are covered elsewhere -- the first by the key's own
    /// field, the second because it is a constant `true` and a change to it is a
    /// format change that must bump `SNAPSHOT_FORMAT_VERSION`.
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
    ///
    /// The window moves between requests -- a sacrificial turn runs in a smaller
    /// one -- so a slot built for the wrong `n_ctx` would name a file whose cache
    /// has a different shape entirely.
    pub(super) fn matches(&self, model_path: &Path, n_ctx: u32) -> bool {
        self.key.n_ctx == n_ctx && self.key.model_path == model_path
    }

    pub(super) fn new(dir: PathBuf, key: SnapshotKey) -> Self {
        let slot = Self {
            dir,
            key,
            stable: None,
            settled: false,
            observations: 0,
            intersected: false,
        };
        let path = slot.path();
        tracing::info!(
            target: KV_TARGET,
            event = "created",
            model = %slot.key.model_path.display(),
            n_ctx = slot.key.n_ctx,
            file_exists = path.exists(),
            path = %path.display(),
            "kv snapshot slot created"
        );
        slot
    }

    pub(super) fn path(&self) -> PathBuf {
        self.dir.join(self.key.file_name())
    }

    /// Fold one prompt into the stable prefix.
    ///
    /// Called for every prompt, including ones that go on to be sacrificial or
    /// to fail: what is being learned is what prompts for this model have in
    /// common, and a prompt that was expensive to serve is exactly as
    /// informative as one that was cheap.
    pub(super) fn observe(&mut self, prompt: &[LlamaToken]) {
        self.observations += 1;
        match &mut self.stable {
            None => self.stable = Some(prompt.to_vec()),
            Some(stable) => {
                let shared = common_prefix_len(stable, prompt);
                if shared < stable.len() {
                    self.intersected = true;
                }
                stable.truncate(shared);
            }
        }
    }

    /// How much of the stable prefix is worth persisting, if a write is due.
    ///
    /// `None` means "not now", for any of three reasons that are deliberately
    /// not distinguished by the caller: a file already exists, too few prompts
    /// have been seen to know what is stable, or the shared run is too short to
    /// pay for itself.
    pub(super) fn writable_prefix(&self) -> Option<usize> {
        if self.settled || !self.intersected {
            return None;
        }
        let len = self.stable.as_ref().map_or(0, Vec::len);
        (len >= SNAPSHOT_MIN_TOKENS).then_some(len)
    }

    /// Persist the first `prefix_len` positions of `ctx`'s sequence 0.
    ///
    /// The caller must have already trimmed the cache to exactly that length —
    /// llama.cpp writes whatever the sequence holds, so trimming is how the
    /// "only the stable prefix" property is enforced rather than merely
    /// intended. `tokens` must be the tokens at those positions.
    ///
    /// Returns whether a file now exists. Failure is logged and swallowed: a
    /// pond that cannot write a cache file still has to answer the turn.
    pub(super) fn write(&mut self, ctx: &LlamaContext<'_>, tokens: &[LlamaToken]) -> bool {
        debug_assert!(
            tokens.len() >= SNAPSHOT_MIN_TOKENS,
            "writable_prefix is the only gate that should reach here"
        );
        let started = std::time::Instant::now();
        if let Err(e) = std::fs::create_dir_all(&self.dir) {
            tracing::warn!(
                target: KV_TARGET,
                event = "discarded",
                reason = "cache directory could not be created",
                error = %e,
                dir = %self.dir.display(),
                "kv snapshot discarded"
            );
            self.settled = true;
            return false;
        }

        // Written beside the target and renamed, so a process killed mid-write
        // cannot leave a truncated file that the next start would open as valid.
        // llama.cpp's loader does check its own header, but a torn file is a
        // failure mode worth removing rather than relying on someone else's
        // validation to catch.
        let final_path = self.path();
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
                    self.settled = true;
                    return false;
                }
                tracing::info!(
                    target: KV_TARGET,
                    event = "saved",
                    tokens = tokens.len(),
                    bytes,
                    kib_per_token = bytes as f64 / 1024.0 / tokens.len().max(1) as f64,
                    elapsed_ms = started.elapsed().as_millis() as u64,
                    model = %self.key.model_path.display(),
                    n_ctx = self.key.n_ctx,
                    path = %final_path.display(),
                    "kv snapshot saved; cold starts for this model now resume from disk"
                );
                self.settled = true;
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
                // Settled even on failure: a slot that cannot write once will
                // not write on the tenth attempt either, and retrying would put
                // a multi-hundred-megabyte write in front of every turn.
                self.settled = true;
                false
            }
        }
    }

    /// Restore a snapshot into `ctx` and report how much of `prompt` it covers.
    ///
    /// `None` leaves the context untouched and means "prefill normally".
    ///
    /// The loaded tokens are checked against `prompt` rather than trusted: the
    /// file-name key cannot see a changed system prompt or a changed tool set, both
    /// of which alter the preamble without touching the model. What makes this
    /// safe is that a mismatch is cheap — the restored cache is cleared and the
    /// turn prefills as it would have anyway.
    pub(super) fn load(
        &mut self,
        ctx: &mut LlamaContext<'_>,
        prompt: &[LlamaToken],
    ) -> Option<usize> {
        let path = self.path();
        if !path.exists() {
            return None;
        }
        let started = std::time::Instant::now();
        // An existing file settles the slot whatever happens next: if it loads,
        // there is nothing to write; if it does not, writing over it is the job
        // of a later process that can observe a stable prefix again.
        self.settled = true;

        // The context's capacity, NOT the prompt length. `max_tokens` sizes the
        // output buffer llama.cpp writes the token list into, so passing the
        // prompt length makes any snapshot LONGER than the current prompt fail
        // outright with "token count in sequence state file exceeded capacity".
        // That is not the check it looks like: a longer snapshot is a perfectly
        // ordinary thing to find and belongs in the prefix test below, where it
        // can be rejected in a way that says why.
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
                self.retire(&path, "unreadable");
                clear_seq(ctx);
                return None;
            }
        };

        // The tokens must be a PREFIX of this prompt, not merely similar to it.
        // Anything less and the KV holds attention state for tokens the prompt
        // does not contain, which is not a slow answer -- it is a wrong one.
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
            // A snapshot this pond's prompts do not begin with is not going to
            // start matching, and leaving it costs a failed read on EVERY cold
            // start for the life of the install. Retiring it lets a shorter,
            // genuinely shared prefix take its place once one is known -- which
            // is how a first write that over-fitted one session corrects itself.
            if shared < tokens.len() {
                self.retire(&path, "this pond's prompts do not begin with it");
            }
            clear_seq(ctx);
            return None;
        }

        tracing::info!(
            target: KV_TARGET,
            event = "loaded",
            reused_tokens = shared,
            prompt_tokens = prompt.len(),
            bytes,
            elapsed_ms = started.elapsed().as_millis() as u64,
            model = %self.key.model_path.display(),
            "kv snapshot loaded; the preamble prefill is skipped"
        );
        Some(shared)
    }
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

impl SnapshotSlot {
    /// Delete a snapshot that cannot serve this pond, and allow a replacement.
    ///
    /// Clearing `settled` is the half that matters: without it the slot has
    /// already decided not to write this process, so a bad file would be deleted
    /// and never replaced, and the pond would go back to paying a full cold
    /// prefill with nothing on disk and nothing trying to put anything there.
    fn retire(&mut self, path: &Path, reason: &str) {
        match std::fs::remove_file(path) {
            Ok(()) => tracing::info!(
                target: KV_TARGET,
                event = "discarded",
                reason = %reason,
                scope = "file deleted; a replacement may be written once a shared prefix is known",
                path = %path.display(),
                "kv snapshot retired"
            ),
            Err(e) => tracing::debug!(
                target: KV_TARGET,
                error = %e,
                path = %path.display(),
                "kv snapshot could not be retired"
            ),
        }
        self.settled = false;
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

    fn tok(ids: &[i32]) -> Vec<LlamaToken> {
        ids.iter().copied().map(LlamaToken).collect()
    }

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

    /// The stable prefix is an intersection, not a memory of the last prompt.
    /// A snapshot built from one prompt would contain that prompt's user
    /// message, which is the privacy property this type exists to hold.
    #[test]
    fn the_stable_prefix_shrinks_to_what_every_prompt_shares() {
        let mut s = slot();
        s.observe(&tok(&[1, 2, 3, 4, 5]));
        assert_eq!(s.stable.as_deref(), Some(tok(&[1, 2, 3, 4, 5]).as_slice()));

        s.observe(&tok(&[1, 2, 3, 9, 9]));
        assert_eq!(s.stable.as_deref(), Some(tok(&[1, 2, 3]).as_slice()));

        // Never grows back, even if a later prompt shares more with the
        // previous one than the intersection holds.
        s.observe(&tok(&[1, 2, 3, 9, 9]));
        assert_eq!(s.stable.as_deref(), Some(tok(&[1, 2, 3]).as_slice()));
    }

    #[test]
    fn a_prompt_sharing_nothing_collapses_the_prefix_to_empty() {
        let mut s = slot();
        s.observe(&tok(&[1, 2, 3]));
        s.observe(&tok(&[7, 8, 9]));
        assert_eq!(s.stable.as_deref(), Some([].as_slice()));
        assert_eq!(s.writable_prefix(), None);
    }

    /// A short shared run is not worth a file. `REUSE_MIN_TOKENS` governs a free
    /// in-memory resume; this threshold governs hundreds of megabytes of IO.
    #[test]
    fn a_prefix_below_the_threshold_is_not_written() {
        // Both cases intersect first, so the threshold is what is under test
        // rather than the intersection requirement beside it.
        let mut s = slot();
        let short: Vec<i32> = (0..(SNAPSHOT_MIN_TOKENS as i32 - 1)).collect();
        s.observe(&tok(&short));
        s.observe(&tok(&short[..short.len() - 1]));
        assert_eq!(s.writable_prefix(), None);

        let mut s = slot();
        let long: Vec<i32> = (0..(SNAPSHOT_MIN_TOKENS as i32 + 50)).collect();
        s.observe(&tok(&long));
        s.observe(&tok(&long[..SNAPSHOT_MIN_TOKENS]));
        assert_eq!(s.writable_prefix(), Some(SNAPSHOT_MIN_TOKENS));
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
        let long: Vec<i32> = (0..(SNAPSHOT_MIN_TOKENS as i32 + 200)).collect();

        // Three prompts, all identical: the count is met and nothing has been
        // learned. This is a whole conversation's worth of turns.
        let mut s = slot();
        for _ in 0..3 {
            s.observe(&tok(&long));
        }
        assert_eq!(
            s.writable_prefix(),
            None,
            "agreeing prompts prove nothing about what a DIFFERENT session shares"
        );

        // One genuinely different prompt, and now the prefix is an intersection.
        let mut diverged = long.clone();
        let cut = SNAPSHOT_MIN_TOKENS + 100;
        diverged[cut] = -1;
        s.observe(&tok(&diverged));
        assert_eq!(
            s.writable_prefix(),
            Some(cut),
            "a shrink is the evidence, and the shared run is what gets written"
        );
    }

    /// Writing is once per slot. Repeating it would put a multi-hundred-megabyte
    /// write in front of a turn, every turn, for a file that is already there.
    #[test]
    fn a_settled_slot_never_offers_another_write() {
        let mut s = slot();
        let long: Vec<i32> = (0..(SNAPSHOT_MIN_TOKENS as i32 + 50)).collect();
        s.observe(&tok(&long));
        s.observe(&tok(&long[..SNAPSHOT_MIN_TOKENS]));
        assert!(s.writable_prefix().is_some());
        // Set directly: this is the state `load` and `write` both leave behind,
        // and the assertion is about what `writable_prefix` does with it.
        s.settled = true;
        assert_eq!(s.writable_prefix(), None);
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
        let name = base.file_name();

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
                v.file_name(),
                name,
                "changing a key component must change the file name: {v:?}"
            );
        }
        // ...and the same key is the same name, or nothing is ever reused.
        assert_eq!(base.file_name(), name);
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
            .file_name()
            .starts_with(&format!("v{SNAPSHOT_FORMAT_VERSION}-")));
        assert!(k.file_name().ends_with(".kv"));
    }
}
