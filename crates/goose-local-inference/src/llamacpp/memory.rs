//! Unified memory on a Jetson: what loading a model leaves in the page cache, how much memory the
//! GPU can really have, and how large a context that buys.
//!
//! On Tegra (Jetson Orin) the GPU allocates from system memory through nvmap, and nvmap fails a
//! `cudaMalloc` when *free* memory runs low: it does not reclaim page cache to satisfy one.
//! llama.cpp copies the GPU-resident weights out of the mmapped GGUF, so a load fills the page
//! cache with the very bytes it has just copied. On 2026-10-27 Gemma 4 E4B's 2.2 GB of GPU weights
//! loaded, the cache grew by 2.2 GB, 251 MB stayed free, and a 42 MiB allocation (the drafter's,
//! by its place in the log) and the KV cache's 256 MiB both failed. llama.cpp reports
//! `MemAvailable` as a UMA device's
//! free memory, which counts that cache, so its own estimate cannot see the problem.
//!
//! The remedy here: release the model's file from the page cache as the load copies it and once
//! it is done ([`release_files`], [`LoadProgressRelease`]), and size the context against
//! `MemFree` measured after that release ([`MemoryFit`]).

use std::path::{Path, PathBuf};

use super::super::load_error::{LoadStage, ModelLoadError};

const MIB: u64 = 1024 * 1024;

/// How much of a load runs between two releases of its file from the page cache. Small enough
/// that the cache never holds more than this of the copied weights, large enough that a 3 GB
/// model is released about a dozen times rather than once per tensor.
pub(super) const RELEASE_STEP_BYTES: u64 = 256 * MIB;

/// Context windows tried, widest first, when the requested one does not fit.
pub(super) const CONTEXT_LADDER: [usize; 5] = [16_384, 12_288, 8_192, 6_144, 4_096];

/// The narrowest window the fit hands out, whatever free memory says. Below it a Gemma 4 E4B
/// turn (a ~4,500-token preamble) cannot run at all, so a context this small that then fails to
/// allocate is reported as out of memory rather than shrunk further.
pub(super) const CONTEXT_FLOOR: usize = 4_096;

/// Free memory a context leaves untouched: the drafter's context (+48 MB measured on the Orin),
/// a sacrificial side-call context, and nvmap's own slack (a 42 MiB allocation failed with
/// 251 MB free on 2026-10-27).
pub(super) const SAFETY_MARGIN_BYTES: u64 = 512 * MIB;

/// llama.cpp's compute buffers for one context. NOT an Orin measurement with logits reserved
/// for one output (`with_n_outputs_max`); the 522 MiB measured before that change was mostly a
/// full vocabulary row per ubatch token. Replace with the `compute buffer size` llama.cpp logs on
/// the Orin.
pub(super) const COMPUTE_RESERVE_BYTES: u64 = 256 * MIB;

fn mib(bytes: u64) -> u64 {
    bytes / MIB
}

// ── /proc/meminfo ───────────────────────────────────────────────────────────

/// The kernel's memory figures, bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct MemInfo {
    /// `MemFree`: what an nvmap allocation can actually get.
    pub free: u64,
    /// `Cached`: page cache, which nvmap does not reclaim.
    pub cached: u64,
    /// `MemAvailable`: free plus reclaimable, what `free -m` and llama.cpp report.
    pub available: u64,
}

impl MemInfo {
    /// Parse `/proc/meminfo`; `None` unless all three fields are present.
    pub(super) fn parse(text: &str) -> Option<Self> {
        let field = |name: &str| {
            text.lines().find_map(|line| {
                let value = line.strip_prefix(name)?.strip_prefix(':')?;
                let kib = value
                    .trim()
                    .strip_suffix("kB")?
                    .trim()
                    .parse::<u64>()
                    .ok()?;
                kib.checked_mul(1024)
            })
        };
        Some(Self {
            free: field("MemFree")?,
            cached: field("Cached")?,
            available: field("MemAvailable")?,
        })
    }

    /// This machine's figures, or `None` where there is no `/proc/meminfo`.
    pub(super) fn read() -> Option<Self> {
        if cfg!(target_os = "linux") {
            std::fs::read_to_string("/proc/meminfo")
                .ok()
                .and_then(|text| Self::parse(&text))
        } else {
            None
        }
    }
}

// ── Releasing a file from the page cache ────────────────────────────────────

/// Drops what the page cache holds of a file. A trait so the load path can be tested without a
/// kernel that honours the advice.
pub(super) trait PageCache: Send + Sync {
    /// Release `path`, including the pages this process maps read-only. Never changes what a
    /// mapping reads: a released page is read back from the file on its next access.
    fn release(&self, path: &Path) -> std::io::Result<()>;
}

/// The kernel's page cache.
///
/// `posix_fadvise(POSIX_FADV_DONTNEED)` drops a file's cached pages, but skips every page a
/// process has mapped, and llama.cpp maps the whole GGUF (`use_mmap`, kept for the reason in
/// `LlamaCppBackend::load_model`): the weights it copied to the GPU were read through that
/// mapping and stay mapped until the model is freed. So the file's read-only shared mappings in
/// this process are first paged out with `madvise(MADV_PAGEOUT)` (Linux 5.4+), which never
/// discards data on any mapping, and then the file is advised away.
pub(super) struct SystemPageCache;

#[cfg(target_os = "linux")]
impl PageCache for SystemPageCache {
    fn release(&self, path: &Path) -> std::io::Result<()> {
        use std::os::fd::AsRawFd;
        use std::os::unix::fs::MetadataExt;

        let file = std::fs::File::open(path)?;
        let inode = file.metadata()?.ino();
        let canonical = std::fs::canonicalize(path)?;
        let maps = std::fs::read_to_string("/proc/self/maps")?;
        for mapping in maps.lines().filter_map(Mapping::parse) {
            if !mapping.is_read_only_view_of(inode, &canonical) {
                continue;
            }
            // SAFETY: advice only. MADV_PAGEOUT reclaims the range's pages the way memory
            // pressure would: a clean file page is dropped and read back from the file on its
            // next access, an anonymous one is swapped. It cannot change what any mapping reads,
            // so even a range unmapped and reused since `/proc/self/maps` was read is unharmed.
            let rc = unsafe {
                libc::madvise(
                    mapping.start as *mut libc::c_void,
                    mapping.end - mapping.start,
                    libc::MADV_PAGEOUT,
                )
            };
            if rc != 0 {
                // EINVAL on a kernel older than 5.4: the advice below still drops what is not
                // mapped, which is everything llama.cpp has unmapped after the load.
                tracing::debug!(
                    target: "giap::trace",
                    file = %path.display(),
                    error = %std::io::Error::last_os_error(),
                    "madvise(MADV_PAGEOUT) refused; releasing the unmapped pages only"
                );
            }
        }
        // SAFETY: an advisory call on a descriptor this function owns for its whole duration.
        let rc = unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) };
        if rc != 0 {
            return Err(std::io::Error::from_raw_os_error(rc));
        }
        Ok(())
    }
}

#[cfg(not(target_os = "linux"))]
impl PageCache for SystemPageCache {
    fn release(&self, _path: &Path) -> std::io::Result<()> {
        Err(std::io::ErrorKind::Unsupported.into())
    }
}

/// One line of `/proc/self/maps`. Read only where the kernel has one.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
#[derive(Debug, Clone, PartialEq, Eq)]
struct Mapping {
    start: usize,
    end: usize,
    writable: bool,
    shared: bool,
    inode: u64,
    path: PathBuf,
}

#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
impl Mapping {
    /// `start-end perms offset dev inode path`; the path may contain spaces, or be absent.
    fn parse(line: &str) -> Option<Self> {
        let mut rest = line;
        let mut field = || {
            rest = rest.trim_start();
            let end = rest.find(char::is_whitespace).unwrap_or(rest.len());
            let (value, tail) = rest.split_at(end);
            rest = tail;
            (!value.is_empty()).then_some(value)
        };
        let (start, end) = field()?.split_once('-')?;
        let perms = field()?.as_bytes();
        let _offset = field()?;
        let _device = field()?;
        let inode = field()?.parse().ok()?;
        let path = rest.trim();
        if perms.len() < 4 || path.is_empty() {
            return None;
        }
        Some(Self {
            start: usize::from_str_radix(start, 16).ok()?,
            end: usize::from_str_radix(end, 16).ok()?,
            writable: perms[1] == b'w',
            shared: perms[3] == b's',
            inode,
            path: PathBuf::from(path),
        })
    }

    /// A read-only shared mapping of the file `(inode, path)`: how llama.cpp maps a GGUF.
    fn is_read_only_view_of(&self, inode: u64, path: &Path) -> bool {
        self.inode == inode
            && self.path == path
            && self.shared
            && !self.writable
            && self.end > self.start
    }
}

/// Release `files` from the page cache and say what that freed, logged as
/// `kind="page_cache_released"`. Returns the free memory after, when the kernel reports it.
pub(super) fn release_files(
    cache: &dyn PageCache,
    files: &[&Path],
    reason: &'static str,
) -> Option<MemInfo> {
    let before = MemInfo::read();
    let mut released = 0usize;
    for file in files {
        match cache.release(file) {
            Ok(()) => released += 1,
            Err(e) => tracing::warn!(
                target: "giap::trace",
                kind = "page_cache_release_failed",
                reason,
                file = %file.display(),
                error = %e,
                "could not release a model file from the page cache"
            ),
        }
    }
    let after = MemInfo::read();
    tracing::info!(
        target: "giap::trace",
        kind = "page_cache_released",
        reason,
        files = released,
        free_mb_before = before.map(|m| mib(m.free)),
        free_mb_after = after.map(|m| mib(m.free)),
        cached_mb_before = before.map(|m| mib(m.cached)),
        cached_mb_after = after.map(|m| mib(m.cached)),
        "released model files from the page cache"
    );
    after
}

/// Releases a file while llama.cpp loads it, from the load's progress callback, so the cache never
/// fills with the weights already copied to the GPU: once per [`RELEASE_STEP_BYTES`] of progress.
pub(super) struct LoadProgressRelease {
    cache: std::sync::Arc<dyn PageCache>,
    path: PathBuf,
    file_len: u64,
    released_at: u64,
    releases: u32,
}

impl LoadProgressRelease {
    pub(super) fn new(cache: std::sync::Arc<dyn PageCache>, path: &Path) -> Self {
        let file_len = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        Self {
            cache,
            path: path.to_path_buf(),
            file_len,
            released_at: 0,
            releases: 0,
        }
    }

    /// llama.cpp's progress, `0.0..=1.0` of the bytes loaded. Never aborts the load: a release
    /// that fails is logged at debug and the load goes on, released or not.
    pub(super) fn on_progress(&mut self, progress: f32) -> bool {
        let loaded = (f64::from(progress.clamp(0.0, 1.0)) * self.file_len as f64) as u64;
        let steps = loaded.saturating_sub(self.released_at) / RELEASE_STEP_BYTES;
        if steps > 0 {
            // On the step boundary, not at `loaded`, so coarse progress does not drift.
            self.released_at += steps * RELEASE_STEP_BYTES;
            self.releases += 1;
            if let Err(e) = self.cache.release(&self.path) {
                tracing::debug!(
                    target: "giap::trace",
                    file = %self.path.display(),
                    error = %e,
                    "page cache release during the load failed"
                );
            }
        }
        true
    }

    #[cfg(test)]
    fn releases(&self) -> u32 {
        self.releases
    }
}

// ── The context free memory buys ────────────────────────────────────────────

/// What a KV cache costs, read from the model's own metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct KvGeometry {
    /// Layers that allocate KV: all of them less `<arch>.attention.shared_kv_layers`.
    pub owning_layers: u64,
    pub kv_heads: u64,
    pub key_length: u64,
    pub value_length: u64,
}

impl KvGeometry {
    /// The geometry `model` declares; `None` when a count it needs is zero. KV heads are layer
    /// 0's, as llama.cpp reports them; key and value widths default to the head dimension.
    pub(super) fn of(model: &llama_cpp_2::model::LlamaModel) -> Option<Self> {
        let arch = model.meta_val_str("general.architecture").ok()?;
        let meta = |key: &str| {
            model
                .meta_val_str(&format!("{arch}.{key}"))
                .ok()
                .and_then(|value| value.trim().parse::<u64>().ok())
        };
        let layers = u64::from(model.n_layer());
        let n_head = u64::from(model.n_head());
        let kv_heads = u64::from(model.n_head_kv());
        let n_embd = u64::try_from(model.n_embd()).ok()?;
        if layers == 0 || n_head == 0 || kv_heads == 0 {
            return None;
        }
        let head_dim = n_embd / n_head;
        let owning_layers = layers.saturating_sub(meta("attention.shared_kv_layers").unwrap_or(0));
        (owning_layers > 0).then(|| Self {
            owning_layers,
            kv_heads,
            key_length: meta("attention.key_length").unwrap_or(head_dim),
            value_length: meta("attention.value_length").unwrap_or(head_dim),
        })
    }

    /// Bytes of KV cache per token of context.
    ///
    /// Charges every owning layer the full window at the full key width, at least what
    /// llama.cpp's full-size cache costs. Contexts keep the windowed cache (`swa_full = false`),
    /// whose sliding-window layers hold only their window: on the Orin E4B's 16K cache was
    /// 476 MiB full-size and 152 MiB windowed (q8_0). The header does not say which layers
    /// slide, so the bound errs on the side that cannot run out of memory.
    pub(super) fn bytes_per_token(&self, type_k: Option<&str>, type_v: Option<&str>) -> u64 {
        let per_head =
            self.key_length * bytes_per_32(type_k) + self.value_length * bytes_per_32(type_v);
        (self.owning_layers * self.kv_heads * per_head).div_ceil(32)
    }
}

/// Bytes per 32 cache elements for a ggml KV type name; f16 when unset or unknown, which is
/// what `build_context_params` falls back to.
fn bytes_per_32(type_name: Option<&str>) -> u64 {
    match type_name.map(str::to_ascii_lowercase).as_deref() {
        Some("f32") => 128,
        Some("q8_0") => 34,
        Some("q5_1") => 24,
        Some("q5_0") => 22,
        Some("q4_1") => 20,
        Some("q4_0") | Some("iq4_nl") => 18,
        _ => 64,
    }
}

/// Why a window came out the size it did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum FitReason {
    /// The requested window fits.
    Requested,
    /// A narrower rung of [`CONTEXT_LADDER`] fits; the requested one does not.
    Reduced,
    /// Nothing at or above [`CONTEXT_FLOOR`] fits; the floor is handed out anyway.
    Floor,
}

impl FitReason {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::Requested => "requested_fits",
            Self::Reduced => "reduced_to_fit",
            Self::Floor => "floor",
        }
    }
}

/// A window and why.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ContextFit {
    pub n_ctx: usize,
    pub reason: FitReason,
}

/// The widest window, at most `requested`, whose KV cache fits `budget` bytes: `requested` itself,
/// else the widest narrower rung of [`CONTEXT_LADDER`], else [`CONTEXT_FLOOR`] (or `requested`
/// when that is smaller still). Never widens a window.
pub(super) fn fit_context(requested: usize, kv_bytes_per_token: u64, budget: u64) -> ContextFit {
    let fits = |n_ctx: usize| (n_ctx as u64).saturating_mul(kv_bytes_per_token) <= budget;
    if fits(requested) {
        return ContextFit {
            n_ctx: requested,
            reason: FitReason::Requested,
        };
    }
    match CONTEXT_LADDER
        .iter()
        .copied()
        .find(|&rung| rung < requested && rung >= CONTEXT_FLOOR && fits(rung))
    {
        Some(n_ctx) => ContextFit {
            n_ctx,
            reason: FitReason::Reduced,
        },
        None => ContextFit {
            n_ctx: requested.min(CONTEXT_FLOOR),
            reason: FitReason::Floor,
        },
    }
}

/// The memory one loaded model may spend on a context, fixed when it loads.
///
/// Measured once, after the weights and the drafter are in and their files released, so every
/// request for the same window gets the same answer: a window that moved with free memory would
/// rebuild the retained context and change the preamble snapshot's key turn by turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct MemoryFit {
    pub kv_bytes_per_token: u64,
    /// `MemFree` at load, less the picture encoder's weights when the model has one.
    pub free_bytes: u64,
}

impl MemoryFit {
    /// Bytes a KV cache may take: free memory less the compute reserve and the safety margin.
    pub(super) fn kv_budget(&self) -> u64 {
        self.free_bytes
            .saturating_sub(COMPUTE_RESERVE_BYTES)
            .saturating_sub(SAFETY_MARGIN_BYTES)
    }

    pub(super) fn fit(&self, requested: usize) -> ContextFit {
        fit_context(requested, self.kv_bytes_per_token, self.kv_budget())
    }
}

/// Classify an allocation that failed: out of GPU memory when free memory, read now, is short of
/// `needed_mb` plus the safety margin. `None` (some other failure) when free memory is plenty or
/// cannot be read.
pub(super) fn out_of_gpu_memory(
    stage: LoadStage,
    needed_mb: u64,
    now: Option<MemInfo>,
) -> Option<ModelLoadError> {
    let free_mb = mib(now?.free);
    (free_mb < needed_mb.saturating_add(mib(SAFETY_MARGIN_BYTES))).then_some(
        ModelLoadError::OutOfGpuMemory {
            stage,
            needed_mb,
            free_mb,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    const ORIN_MEMINFO: &str = "MemTotal:        7802364 kB\n\
                                MemFree:          257024 kB\n\
                                MemAvailable:    4561920 kB\n\
                                Buffers:           12340 kB\n\
                                Cached:          4194304 kB\n\
                                SwapCached:            0 kB\n";

    #[test]
    fn meminfo_reads_free_cached_and_available_in_bytes() {
        let info = MemInfo::parse(ORIN_MEMINFO).expect("parses");
        assert_eq!(info.free, 257_024 * 1024);
        assert_eq!(info.cached, 4_194_304 * 1024, "Cached, not SwapCached");
        assert_eq!(info.available, 4_561_920 * 1024);
    }

    #[test]
    fn meminfo_without_a_field_is_unreadable() {
        assert_eq!(
            MemInfo::parse("MemTotal: 1 kB\nMemFree: 1 kB\nCached: 1 kB\n"),
            None
        );
        assert_eq!(
            MemInfo::parse("MemFree: lots\nCached: 1 kB\nMemAvailable: 1 kB\n"),
            None
        );
        assert_eq!(MemInfo::parse(""), None);
    }

    #[test]
    fn a_maps_line_names_the_file_and_its_permissions() {
        let line = "ffff8a000000-ffff96000000 r--s 00000000 103:02 1234567    \
                    /home/giap/models/gguf/gemma 4 E4B.gguf";
        let mapping = Mapping::parse(line).expect("parses");
        assert_eq!(mapping.start, 0xffff_8a00_0000);
        assert_eq!(mapping.end, 0xffff_9600_0000);
        assert!(mapping.shared && !mapping.writable);
        assert_eq!(mapping.inode, 1_234_567);
        assert_eq!(
            mapping.path,
            PathBuf::from("/home/giap/models/gguf/gemma 4 E4B.gguf")
        );
        let path = Path::new("/home/giap/models/gguf/gemma 4 E4B.gguf");
        assert!(mapping.is_read_only_view_of(1_234_567, path));
        assert!(
            !mapping.is_read_only_view_of(7, path),
            "another file at that path"
        );
    }

    #[test]
    fn only_a_read_only_shared_file_mapping_is_released() {
        let path = Path::new("/m.gguf");
        for (line, expected) in [
            ("1000-2000 r--s 0 103:02 9 /m.gguf", true),
            ("1000-2000 rw-s 0 103:02 9 /m.gguf", false),
            ("1000-2000 r--p 0 103:02 9 /m.gguf", false),
            ("1000-2000 r--s 0 103:02 9 /other.gguf", false),
        ] {
            let mapping = Mapping::parse(line).expect("parses");
            assert_eq!(mapping.is_read_only_view_of(9, path), expected, "{line}");
        }
        assert_eq!(
            Mapping::parse("1000-2000 rw-p 00000000 00:00 0"),
            None,
            "anonymous"
        );
        assert_eq!(Mapping::parse("garbage"), None);
    }

    /// Records each release instead of advising a kernel.
    #[derive(Default)]
    struct Recording(Mutex<Vec<PathBuf>>);

    impl PageCache for Recording {
        fn release(&self, path: &Path) -> std::io::Result<()> {
            self.0.lock().unwrap().push(path.to_path_buf());
            Ok(())
        }
    }

    #[test]
    fn every_model_file_is_released() {
        let cache = Recording::default();
        let (target, drafter) = (Path::new("/m/target.gguf"), Path::new("/m/drafter.gguf"));
        release_files(&cache, &[target, drafter], "test");
        assert_eq!(
            *cache.0.lock().unwrap(),
            vec![target.to_path_buf(), drafter.to_path_buf()]
        );
    }

    #[test]
    fn a_load_is_released_once_per_step_of_progress() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.gguf");
        let file = std::fs::File::create(&path).unwrap();
        file.set_len(3 * 1024 * MIB).unwrap();
        let cache = Arc::new(Recording::default());
        let mut release = LoadProgressRelease::new(cache.clone(), &path);
        for i in 0..=1000 {
            assert!(
                release.on_progress(i as f32 / 1000.0),
                "never aborts the load"
            );
        }
        // 3 GiB in steps of 256 MiB.
        assert_eq!(release.releases(), 12);
        assert_eq!(cache.0.lock().unwrap().len(), 12);
        assert!(cache.0.lock().unwrap().iter().all(|p| *p == path));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn the_kernel_accepts_the_release_of_a_mapped_file() {
        use std::io::Write;
        use std::os::fd::AsRawFd;
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(&vec![7u8; 4 * MIB as usize]).unwrap();
        file.flush().unwrap();
        let len = 4 * MIB as usize;
        // SAFETY: a read-only shared mapping of a file this test owns, unmapped below.
        let addr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_SHARED,
                file.as_file().as_raw_fd(),
                0,
            )
        };
        assert_ne!(addr, libc::MAP_FAILED);
        // SAFETY: within the mapping just made.
        let before = unsafe { *(addr as *const u8).add(len - 1) };
        SystemPageCache
            .release(file.path())
            .expect("fadvise and madvise accepted");
        // SAFETY: as above; a released page reads back from the file.
        let after = unsafe { *(addr as *const u8).add(len - 1) };
        assert_eq!((before, after), (7, 7));
        // SAFETY: the mapping made above, unmapped once.
        unsafe { libc::munmap(addr, len) };
    }

    /// An illustrative geometry, not any shipped model's.
    const GEOMETRY: KvGeometry = KvGeometry {
        owning_layers: 24,
        kv_heads: 2,
        key_length: 512,
        value_length: 512,
    };

    #[test]
    fn kv_cost_follows_the_cache_types() {
        let f16 = GEOMETRY.bytes_per_token(None, None);
        assert_eq!(f16, 24 * 2 * (512 + 512) * 2);
        assert_eq!(GEOMETRY.bytes_per_token(Some("F16"), Some("f16")), f16);
        assert_eq!(
            GEOMETRY.bytes_per_token(Some("q8_0"), Some("q8_0")),
            f16 * 34 / 64
        );
        assert_eq!(GEOMETRY.bytes_per_token(Some("f32"), None), f16 * 3 / 2);
        assert_eq!(
            GEOMETRY.bytes_per_token(Some("mystery"), None),
            f16,
            "unknown is f16"
        );
    }

    #[test]
    fn the_requested_window_is_kept_when_it_fits() {
        let per_token = GEOMETRY.bytes_per_token(None, None);
        let fit = fit_context(16_384, per_token, 16_384 * per_token);
        assert_eq!(
            fit,
            ContextFit {
                n_ctx: 16_384,
                reason: FitReason::Requested
            }
        );
        // Off the ladder, and above it, alike.
        assert_eq!(fit_context(10_240, per_token, u64::MAX).n_ctx, 10_240);
        assert_eq!(fit_context(131_072, per_token, u64::MAX).n_ctx, 131_072);
    }

    #[test]
    fn a_window_that_does_not_fit_steps_down_the_ladder() {
        let per_token = 100;
        assert_eq!(
            fit_context(16_384, per_token, 12_288 * per_token),
            ContextFit {
                n_ctx: 12_288,
                reason: FitReason::Reduced
            }
        );
        assert_eq!(
            fit_context(16_384, per_token, 12_287 * per_token).n_ctx,
            8_192
        );
        assert_eq!(
            fit_context(10_240, per_token, 9_000 * per_token).n_ctx,
            8_192
        );
        assert_eq!(
            fit_context(16_384, per_token, 6_144 * per_token).n_ctx,
            6_144
        );
    }

    #[test]
    fn the_floor_holds_whatever_free_memory_says() {
        assert_eq!(
            fit_context(16_384, 100, 0),
            ContextFit {
                n_ctx: CONTEXT_FLOOR,
                reason: FitReason::Floor
            }
        );
        assert_eq!(fit_context(16_384, 100, 4_095 * 100).n_ctx, CONTEXT_FLOOR);
        // A window already under the floor is never widened to it.
        assert_eq!(fit_context(2_048, 100, 0).n_ctx, 2_048);
    }

    #[test]
    fn the_budget_keeps_the_compute_reserve_and_the_margin() {
        let per_token = GEOMETRY.bytes_per_token(None, None);
        let exactly = 16_384 * per_token + COMPUTE_RESERVE_BYTES + SAFETY_MARGIN_BYTES;
        let fit = MemoryFit {
            kv_bytes_per_token: per_token,
            free_bytes: exactly,
        };
        assert_eq!(fit.fit(16_384).reason, FitReason::Requested);
        let short = MemoryFit {
            free_bytes: exactly - 1,
            ..fit
        };
        assert_eq!(
            short.fit(16_384),
            ContextFit {
                n_ctx: 12_288,
                reason: FitReason::Reduced
            }
        );
    }

    /// The figures of 2026-10-27: the context failed with 251 MB free, the weights with ~1 GB.
    #[test]
    fn a_failure_is_out_of_memory_only_when_free_memory_is_short() {
        let now = |free_mb: u64| {
            Some(MemInfo {
                free: free_mb * MIB,
                cached: 0,
                available: free_mb * MIB,
            })
        };
        assert_eq!(
            out_of_gpu_memory(LoadStage::Context, 542, now(251)),
            Some(ModelLoadError::OutOfGpuMemory {
                stage: LoadStage::Context,
                needed_mb: 542,
                free_mb: 251
            })
        );
        assert_eq!(
            out_of_gpu_memory(LoadStage::Weights, 3_052, now(1_024)).map(|e| e.to_string()),
            Some(
                "out of GPU memory for the model weights: needs about 3052 MiB, 1024 MiB free"
                    .to_string()
            )
        );
        assert_eq!(
            out_of_gpu_memory(LoadStage::Context, 542, now(4_700)),
            None,
            "plenty free"
        );
        assert_eq!(
            out_of_gpu_memory(LoadStage::Context, 542, None),
            None,
            "unreadable"
        );
    }
}
