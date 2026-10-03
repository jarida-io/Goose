//! Runtime binding for the LiteRT-LM C API 1.0.0 (`c/engine.h`,
//! `c/conversation.h`, `c/model_info.h`, `c/error_reporter.h`), plus, when the
//! library has them, the KV snapshot and prompt-prefill functions of GIAP's
//! fork, giap-main.
//!
//! The library carries its own Rust std, allocator, abseil and protobuf, so it
//! is opened by absolute path with `RTLD_LOCAL` and only C types cross the
//! boundary. Every call returns a `LiteRtLmStatusCode`; on failure the message
//! is read from the calling thread's last-error slot straight away.

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::ptr::{self, NonNull};
use std::sync::{mpsc, Mutex, OnceLock};
use std::time::Duration;

use goose_provider_types::errors::ProviderError;
use libloading::Library;

use crate::paths::Paths;

pub(super) const LIB_DIR_ENV: &str = "GOOSE_LITERT_LIB_DIR";
const BUNDLE_SUBDIR: &str = "litert-lm";
const DATA_SUBDIR: &str = "litert-lm/lib";
#[cfg(target_os = "macos")]
const LIB_FILE: &str = "liblitert-lm.dylib";
#[cfg(windows)]
const LIB_FILE: &str = "litert-lm.dll";
#[cfg(not(any(target_os = "macos", windows)))]
const LIB_FILE: &str = "liblitert-lm.so";
const LOAD_TIMEOUT: Duration = Duration::from_secs(30);

type Status = c_int;
const STATUS_OK: Status = 0;
/// `kLiteRtLmSamplerTypeTopP`: top-k, then top-p. The only type the CPU sampler implements.
const SAMPLER_TOP_P: c_int = 2;

macro_rules! opaque_handles {
    ($($name:ident),* $(,)?) => {
        $(
            #[repr(C)]
            struct $name {
                _private: [u8; 0],
            }
        )*
    };
}

opaque_handles!(
    RawLoadedFile,
    RawEngineSettings,
    RawEngine,
    RawSessionConfig,
    RawSamplerParams,
    RawThinkingConfig,
    RawConversationConfig,
    RawConversation,
    RawBenchmarkInfo,
    RawStreamChunk,
);

type StreamCallback = unsafe extern "C" fn(*mut c_void, *const RawStreamChunk);

/// The C functions this backend calls. Each field resolves `litert_lm_<field>`.
struct Api {
    get_last_error_message: unsafe extern "C" fn() -> *const c_char,
    loaded_file_create: unsafe extern "C" fn(*const c_char, *mut *mut RawLoadedFile) -> Status,
    loaded_file_delete: unsafe extern "C" fn(*mut RawLoadedFile),
    loaded_file_has_speculative_decoding_support:
        unsafe extern "C" fn(*const RawLoadedFile, *mut bool) -> Status,
    loaded_file_max_context_tokens: unsafe extern "C" fn(*const RawLoadedFile, *mut u32) -> Status,
    engine_settings_create: unsafe extern "C" fn(
        *const c_char,
        *const c_char,
        *const c_char,
        *const c_char,
        *mut *mut RawEngineSettings,
    ) -> Status,
    engine_settings_delete: unsafe extern "C" fn(*mut RawEngineSettings),
    engine_settings_set_max_num_tokens:
        unsafe extern "C" fn(*mut RawEngineSettings, c_int) -> Status,
    engine_settings_set_num_threads: unsafe extern "C" fn(*mut RawEngineSettings, c_int) -> Status,
    engine_settings_set_cache_dir:
        unsafe extern "C" fn(*mut RawEngineSettings, *const c_char) -> Status,
    engine_settings_enable_benchmark: unsafe extern "C" fn(*mut RawEngineSettings) -> Status,
    engine_settings_set_enable_speculative_decoding:
        unsafe extern "C" fn(*mut RawEngineSettings, bool) -> Status,
    engine_create: unsafe extern "C" fn(*const RawEngineSettings, *mut *mut RawEngine) -> Status,
    engine_delete: unsafe extern "C" fn(*mut RawEngine),
    session_config_create: unsafe extern "C" fn(*mut *mut RawSessionConfig) -> Status,
    session_config_delete: unsafe extern "C" fn(*mut RawSessionConfig),
    session_config_set_max_output_tokens:
        unsafe extern "C" fn(*mut RawSessionConfig, c_int) -> Status,
    session_config_set_sampler_params:
        unsafe extern "C" fn(*mut RawSessionConfig, *const RawSamplerParams) -> Status,
    sampler_params_create: unsafe extern "C" fn(c_int, *mut *mut RawSamplerParams) -> Status,
    sampler_params_delete: unsafe extern "C" fn(*mut RawSamplerParams),
    sampler_params_set_top_k: unsafe extern "C" fn(*mut RawSamplerParams, i32) -> Status,
    sampler_params_set_top_p: unsafe extern "C" fn(*mut RawSamplerParams, f32) -> Status,
    sampler_params_set_temperature: unsafe extern "C" fn(*mut RawSamplerParams, f32) -> Status,
    sampler_params_set_seed: unsafe extern "C" fn(*mut RawSamplerParams, i32) -> Status,
    thinking_config_create: unsafe extern "C" fn(*mut *mut RawThinkingConfig) -> Status,
    thinking_config_delete: unsafe extern "C" fn(*mut RawThinkingConfig),
    thinking_config_set_enable_thinking:
        unsafe extern "C" fn(*mut RawThinkingConfig, bool) -> Status,
    conversation_config_create: unsafe extern "C" fn(*mut *mut RawConversationConfig) -> Status,
    conversation_config_delete: unsafe extern "C" fn(*mut RawConversationConfig),
    conversation_config_set_session_config:
        unsafe extern "C" fn(*mut RawConversationConfig, *const RawSessionConfig) -> Status,
    conversation_config_set_system_message:
        unsafe extern "C" fn(*mut RawConversationConfig, *const c_char) -> Status,
    conversation_config_set_tools:
        unsafe extern "C" fn(*mut RawConversationConfig, *const c_char) -> Status,
    conversation_config_set_messages:
        unsafe extern "C" fn(*mut RawConversationConfig, *const c_char) -> Status,
    conversation_config_set_thinking_config:
        unsafe extern "C" fn(*mut RawConversationConfig, *const RawThinkingConfig) -> Status,
    conversation_create: unsafe extern "C" fn(
        *mut RawEngine,
        *mut RawConversationConfig,
        *mut *mut RawConversation,
    ) -> Status,
    conversation_delete: unsafe extern "C" fn(*mut RawConversation),
    conversation_send_message_stream: unsafe extern "C" fn(
        *mut RawConversation,
        *const c_char,
        *const c_char,
        *const c_void,
        StreamCallback,
        *mut c_void,
    ) -> Status,
    conversation_cancel_process: unsafe extern "C" fn(*mut RawConversation) -> Status,
    conversation_render_message_to_string:
        unsafe extern "C" fn(*mut RawConversation, *const c_char, *mut *const c_char) -> Status,
    conversation_get_token_count: unsafe extern "C" fn(*mut RawConversation, *mut c_int) -> Status,
    conversation_get_benchmark_info:
        unsafe extern "C" fn(*mut RawConversation, *mut *mut RawBenchmarkInfo) -> Status,
    stream_chunk_get_text:
        unsafe extern "C" fn(*const RawStreamChunk, *mut *const c_char) -> Status,
    stream_chunk_is_final: unsafe extern "C" fn(*const RawStreamChunk, *mut bool) -> Status,
    stream_chunk_get_error:
        unsafe extern "C" fn(*const RawStreamChunk, *mut *const c_char) -> Status,
    benchmark_info_delete: unsafe extern "C" fn(*mut RawBenchmarkInfo),
    benchmark_info_get_num_prefill_turns:
        unsafe extern "C" fn(*const RawBenchmarkInfo, *mut c_int) -> Status,
    benchmark_info_get_num_decode_turns:
        unsafe extern "C" fn(*const RawBenchmarkInfo, *mut c_int) -> Status,
    benchmark_info_get_prefill_token_count_at:
        unsafe extern "C" fn(*const RawBenchmarkInfo, c_int, *mut c_int) -> Status,
    benchmark_info_get_decode_token_count_at:
        unsafe extern "C" fn(*const RawBenchmarkInfo, c_int, *mut c_int) -> Status,
    benchmark_info_get_prefill_tokens_per_sec_at:
        unsafe extern "C" fn(*const RawBenchmarkInfo, c_int, *mut f64) -> Status,
    /// GIAP's additions, when the library is built from giap-main.
    giap: Option<GiapApi>,
    /// Never unloaded: the function pointers above point into it.
    _library: Library,
}

/// GIAP's giap-main additions to the C API: KV snapshots, and a prompt
/// prefilled as text after the KV cache is rewound, which reuses every token
/// the cache already holds. Upstream LiteRT-LM has none of them; the backend
/// then never parks or rematches a conversation.
struct GiapApi {
    conversation_save_kv_snapshot:
        unsafe extern "C" fn(*mut RawConversation, *const c_char) -> Status,
    conversation_load_kv_snapshot:
        unsafe extern "C" fn(*mut RawConversation, *const c_char) -> Status,
    conversation_send_prefill_text_stream: unsafe extern "C" fn(
        *mut RawConversation,
        c_int,
        *const c_char,
        *const c_char,
        *const c_char,
        *const c_void,
        StreamCallback,
        *mut c_void,
    ) -> Status,
}

impl GiapApi {
    /// All of them or none: a library with only some is not a giap-main build.
    ///
    /// # Safety
    /// As for `symbol`.
    unsafe fn resolve(library: &Library) -> Option<Self> {
        Some(Self {
            conversation_save_kv_snapshot: symbol(
                library,
                "litert_lm_conversation_save_kv_snapshot",
            )
            .ok()?,
            conversation_load_kv_snapshot: symbol(
                library,
                "litert_lm_conversation_load_kv_snapshot",
            )
            .ok()?,
            conversation_send_prefill_text_stream: symbol(
                library,
                "litert_lm_conversation_send_prefill_text_stream",
            )
            .ok()?,
        })
    }
}

macro_rules! resolve_api {
    ($library:ident; $($field:ident),* $(,)?) => {
        Api {
            $( $field: symbol(&$library, concat!("litert_lm_", stringify!($field)))?, )*
            giap: GiapApi::resolve(&$library),
            _library: $library,
        }
    };
}

/// # Safety
/// `T` must be the function-pointer type of the C symbol `name`.
unsafe fn symbol<T: Copy>(library: &Library, name: &str) -> Result<T, String> {
    library
        .get::<T>(name)
        .map(|symbol| *symbol)
        .map_err(|error| format!("missing symbol {name}: {error}"))
}

impl Api {
    /// # Safety
    /// Runs the library's initialisers; `path` must be a LiteRT-LM C API 1.0.0 build.
    unsafe fn load(path: &Path) -> Result<Self, String> {
        let library = open_library(path).map_err(|error| error.to_string())?;
        Ok(resolve_api!(library;
            get_last_error_message,
            loaded_file_create,
            loaded_file_delete,
            loaded_file_has_speculative_decoding_support,
            loaded_file_max_context_tokens,
            engine_settings_create,
            engine_settings_delete,
            engine_settings_set_max_num_tokens,
            engine_settings_set_num_threads,
            engine_settings_set_cache_dir,
            engine_settings_enable_benchmark,
            engine_settings_set_enable_speculative_decoding,
            engine_create,
            engine_delete,
            session_config_create,
            session_config_delete,
            session_config_set_max_output_tokens,
            session_config_set_sampler_params,
            sampler_params_create,
            sampler_params_delete,
            sampler_params_set_top_k,
            sampler_params_set_top_p,
            sampler_params_set_temperature,
            sampler_params_set_seed,
            thinking_config_create,
            thinking_config_delete,
            thinking_config_set_enable_thinking,
            conversation_config_create,
            conversation_config_delete,
            conversation_config_set_session_config,
            conversation_config_set_system_message,
            conversation_config_set_tools,
            conversation_config_set_messages,
            conversation_config_set_thinking_config,
            conversation_create,
            conversation_delete,
            conversation_send_message_stream,
            conversation_cancel_process,
            conversation_render_message_to_string,
            conversation_get_token_count,
            conversation_get_benchmark_info,
            stream_chunk_get_text,
            stream_chunk_is_final,
            stream_chunk_get_error,
            benchmark_info_delete,
            benchmark_info_get_num_prefill_turns,
            benchmark_info_get_num_decode_turns,
            benchmark_info_get_prefill_token_count_at,
            benchmark_info_get_decode_token_count_at,
            benchmark_info_get_prefill_tokens_per_sec_at,
        ))
    }

    /// Turns a status into a `ProviderError` carrying the library's message.
    /// Must run on the thread that made the call, before any other call fails.
    fn check(&self, status: Status, call: &str) -> Result<(), ProviderError> {
        if status == STATUS_OK {
            return Ok(());
        }
        // SAFETY: returns a NUL-terminated thread-local string or NULL.
        let message = unsafe { owned_string((self.get_last_error_message)()) }
            .unwrap_or_else(|| "no error message".to_string());
        Err(ProviderError::ExecutionError(format!(
            "LiteRT-LM {call} failed with {} ({status}): {message}",
            status_name(status)
        )))
    }

    /// Calls a constructor that returns its handle through a trailing out-parameter.
    fn create<T>(
        &self,
        call: &str,
        delete: unsafe extern "C" fn(*mut T),
        construct: impl FnOnce(*mut *mut T) -> Status,
    ) -> Result<Owned<T>, ProviderError> {
        let mut out: *mut T = ptr::null_mut();
        self.check(construct(&mut out), call)?;
        NonNull::new(out)
            .map(|ptr| Owned { ptr, delete })
            .ok_or_else(|| {
                ProviderError::ExecutionError(format!("LiteRT-LM {call} returned no handle"))
            })
    }
}

#[cfg(unix)]
unsafe fn open_library(path: &Path) -> Result<Library, libloading::Error> {
    use libloading::os::unix::{Library as UnixLibrary, RTLD_LOCAL, RTLD_NOW};
    UnixLibrary::open(Some(path), RTLD_NOW | RTLD_LOCAL).map(Library::from)
}

#[cfg(windows)]
unsafe fn open_library(path: &Path) -> Result<Library, libloading::Error> {
    Library::new(path)
}

fn status_name(status: Status) -> &'static str {
    match status {
        1 => "CANCELLED",
        3 => "INVALID_ARGUMENT",
        4 => "DEADLINE_EXCEEDED",
        5 => "NOT_FOUND",
        6 => "ALREADY_EXISTS",
        7 => "PERMISSION_DENIED",
        8 => "RESOURCE_EXHAUSTED",
        9 => "FAILED_PRECONDITION",
        10 => "ABORTED",
        11 => "OUT_OF_RANGE",
        12 => "UNIMPLEMENTED",
        13 => "INTERNAL",
        14 => "UNAVAILABLE",
        15 => "DATA_LOSS",
        16 => "UNAUTHENTICATED",
        _ => "UNKNOWN",
    }
}

/// # Safety
/// `text` must be NULL or point to a NUL-terminated string valid for this call.
unsafe fn owned_string(text: *const c_char) -> Option<String> {
    (!text.is_null()).then(|| CStr::from_ptr(text).to_string_lossy().into_owned())
}

fn c_string(text: &str, what: &str) -> Result<CString, ProviderError> {
    CString::new(text)
        .map_err(|_| ProviderError::ExecutionError(format!("LiteRT-LM {what} contains a NUL byte")))
}

fn path_c_string(path: &Path) -> Result<CString, ProviderError> {
    let text = path.to_str().ok_or_else(|| {
        ProviderError::ExecutionError(format!("Path is not valid UTF-8: {}", path.display()))
    })?;
    c_string(text, "path")
}

fn c_int_saturating(value: usize) -> c_int {
    c_int::try_from(value).unwrap_or(c_int::MAX)
}

static API: OnceLock<Api> = OnceLock::new();
static LOAD_LOCK: Mutex<()> = Mutex::new(());

/// The loaded library, opened on first use. A failed load is not cached, so a
/// library installed later is picked up without a restart.
fn api() -> Result<&'static Api, ProviderError> {
    if let Some(api) = API.get() {
        return Ok(api);
    }
    let _guard = LOAD_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(api) = API.get() {
        return Ok(api);
    }
    let path = locate(&library_dirs())?;
    let api = load_bounded(path)?;
    Ok(API.get_or_init(|| api))
}

/// Opens the library on a helper thread so a hung initialiser or a panic
/// surfaces as an error instead of stalling or unwinding the caller.
fn load_bounded(path: PathBuf) -> Result<Api, ProviderError> {
    let (sender, receiver) = mpsc::channel();
    let thread_path = path.clone();
    std::thread::Builder::new()
        .name("litert-lm-load".to_string())
        .spawn(move || {
            let result = catch_unwind(|| unsafe { Api::load(&thread_path) })
                .unwrap_or_else(|_| Err("panicked while loading".to_string()));
            if let Err(mpsc::SendError(Ok(api))) = sender.send(result) {
                // The caller gave up waiting; the library stays mapped.
                std::mem::forget(api);
            }
        })
        .map_err(|error| {
            ProviderError::ExecutionError(format!("Failed to start the LiteRT-LM loader: {error}"))
        })?;
    match receiver.recv_timeout(LOAD_TIMEOUT) {
        Ok(Ok(api)) => {
            tracing::info!(path = %path.display(), "Loaded LiteRT-LM C API library");
            Ok(api)
        }
        Ok(Err(error)) => Err(ProviderError::ExecutionError(format!(
            "Failed to load LiteRT-LM from {}: {error}",
            path.display()
        ))),
        Err(_) => Err(ProviderError::ExecutionError(format!(
            "Timed out after {}s loading LiteRT-LM from {}",
            LOAD_TIMEOUT.as_secs(),
            path.display()
        ))),
    }
}

/// Search order: `GOOSE_LITERT_LIB_DIR`, `<exe dir>/litert-lm`, `<exe dir>`,
/// then goose's data directory.
fn library_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Some(dir) = std::env::var_os(LIB_DIR_ENV).filter(|dir| !dir.is_empty()) {
        dirs.push(PathBuf::from(dir));
    }
    if let Some(exe_dir) = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(Path::to_path_buf))
    {
        dirs.push(exe_dir.join(BUNDLE_SUBDIR));
        dirs.push(exe_dir);
    }
    dirs.push(Paths::in_data_dir(DATA_SUBDIR));
    dirs
}

/// The first directory holding the library, as an absolute path.
fn locate(dirs: &[PathBuf]) -> Result<PathBuf, ProviderError> {
    for dir in dirs {
        let candidate = dir.join(LIB_FILE);
        if candidate.is_file() {
            tracing::debug!(dir = %dir.display(), "Using LiteRT-LM library directory");
            return Ok(std::fs::canonicalize(&candidate)
                .or_else(|_| std::path::absolute(&candidate))
                .unwrap_or(candidate));
        }
    }
    let searched = dirs
        .iter()
        .map(|dir| dir.display().to_string())
        .collect::<Vec<_>>()
        .join(", ");
    Err(ProviderError::ExecutionError(format!(
        "LiteRT-LM library {LIB_FILE} not found (searched: {searched}). Set {LIB_DIR_ENV} to the \
         directory that holds it."
    )))
}

/// A LiteRT-LM handle freed by its matching `*_delete` function.
struct Owned<T> {
    ptr: NonNull<T>,
    delete: unsafe extern "C" fn(*mut T),
}

impl<T> Owned<T> {
    fn as_ptr(&self) -> *mut T {
        self.ptr.as_ptr()
    }
}

impl<T> Drop for Owned<T> {
    fn drop(&mut self) {
        // SAFETY: the pointer came from the matching create call and is freed once.
        unsafe { (self.delete)(self.ptr.as_ptr()) }
    }
}

/// Capability queries on a `.litertlm` file.
pub(super) struct LoadedFile {
    handle: Owned<RawLoadedFile>,
    api: &'static Api,
}

impl LoadedFile {
    pub(super) fn open(path: &Path) -> Result<Self, ProviderError> {
        let api = api()?;
        let path = path_c_string(path)?;
        let handle = api.create("loaded_file_create", api.loaded_file_delete, |out| unsafe {
            (api.loaded_file_create)(path.as_ptr(), out)
        })?;
        Ok(Self { handle, api })
    }

    pub(super) fn supports_speculative_decoding(&self) -> bool {
        let mut supported = false;
        let status = unsafe {
            (self.api.loaded_file_has_speculative_decoding_support)(
                self.handle.as_ptr(),
                &mut supported,
            )
        };
        status == STATUS_OK && supported
    }

    /// `None` when the file does not declare its maximum context.
    pub(super) fn max_context_tokens(&self) -> Option<u32> {
        let mut tokens = 0u32;
        let status =
            unsafe { (self.api.loaded_file_max_context_tokens)(self.handle.as_ptr(), &mut tokens) };
        (status == STATUS_OK && tokens > 0).then_some(tokens)
    }
}

pub(super) struct EngineConfig<'a> {
    pub(super) model_path: &'a Path,
    pub(super) backend: &'a str,
    pub(super) max_num_tokens: usize,
    pub(super) cache_dir: &'a Path,
    pub(super) num_threads: Option<i32>,
    pub(super) speculative_decoding: bool,
}

/// Sampler parameters; setting them writes every field.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct SamplerConfig {
    pub(super) top_k: i32,
    pub(super) top_p: f32,
    pub(super) temperature: f32,
    pub(super) seed: i32,
}

pub(super) struct ConversationConfig<'a> {
    /// System message content, as JSON.
    pub(super) system: Option<&'a str>,
    /// Tool declarations, as a JSON array.
    pub(super) tools: Option<&'a str>,
    /// Earlier messages, as a JSON array.
    pub(super) preface: Option<&'a str>,
    pub(super) enable_thinking: bool,
    pub(super) max_output_tokens: Option<usize>,
    pub(super) sampler: SamplerConfig,
}

pub(super) struct Engine {
    handle: Owned<RawEngine>,
    api: &'static Api,
}

// SAFETY: LiteRT-LM engines are not tied to the thread that created them, and
// the model slot lock gives one thread at a time access to this handle.
unsafe impl Send for Engine {}

impl Engine {
    /// Blocks while the model loads.
    pub(super) fn create(config: &EngineConfig<'_>) -> Result<Self, ProviderError> {
        let api = api()?;
        let model_path = path_c_string(config.model_path)?;
        let backend = c_string(config.backend, "backend name")?;
        let cache_dir = path_c_string(config.cache_dir)?;
        let settings = api.create(
            "engine_settings_create",
            api.engine_settings_delete,
            |out| unsafe {
                (api.engine_settings_create)(
                    model_path.as_ptr(),
                    backend.as_ptr(),
                    ptr::null(),
                    ptr::null(),
                    out,
                )
            },
        )?;
        let raw = settings.as_ptr();
        unsafe {
            api.check(
                (api.engine_settings_set_max_num_tokens)(
                    raw,
                    c_int_saturating(config.max_num_tokens),
                ),
                "engine_settings_set_max_num_tokens",
            )?;
            if let Some(threads) = config.num_threads {
                api.check(
                    (api.engine_settings_set_num_threads)(raw, threads),
                    "engine_settings_set_num_threads",
                )?;
            }
            api.check(
                (api.engine_settings_set_cache_dir)(raw, cache_dir.as_ptr()),
                "engine_settings_set_cache_dir",
            )?;
            api.check(
                (api.engine_settings_enable_benchmark)(raw),
                "engine_settings_enable_benchmark",
            )?;
            api.check(
                (api.engine_settings_set_enable_speculative_decoding)(
                    raw,
                    config.speculative_decoding,
                ),
                "engine_settings_set_enable_speculative_decoding",
            )?;
        }
        let handle = api.create("engine_create", api.engine_delete, |out| unsafe {
            (api.engine_create)(raw, out)
        })?;
        Ok(Self { handle, api })
    }

    /// Whether the library has giap-main's prompt prefill and KV snapshot
    /// calls, which `Conversation::send_prompt`, `save_kv_snapshot` and
    /// `load_kv_snapshot` need.
    pub(super) fn supports_snapshots(&self) -> bool {
        self.api.giap.is_some()
    }

    pub(super) fn conversation(
        &self,
        config: &ConversationConfig<'_>,
    ) -> Result<Conversation, ProviderError> {
        let api = self.api;
        let session = api.create(
            "session_config_create",
            api.session_config_delete,
            |out| unsafe { (api.session_config_create)(out) },
        )?;
        let sampler = api.create(
            "sampler_params_create",
            api.sampler_params_delete,
            |out| unsafe { (api.sampler_params_create)(SAMPLER_TOP_P, out) },
        )?;
        let thinking = api.create(
            "thinking_config_create",
            api.thinking_config_delete,
            |out| unsafe { (api.thinking_config_create)(out) },
        )?;
        let conversation_config = api.create(
            "conversation_config_create",
            api.conversation_config_delete,
            |out| unsafe { (api.conversation_config_create)(out) },
        )?;
        let system = config
            .system
            .map(|system| c_string(system, "system message"))
            .transpose()?;
        let tools = config
            .tools
            .map(|tools| c_string(tools, "tool declarations"))
            .transpose()?;
        let preface = config
            .preface
            .map(|preface| c_string(preface, "message history"))
            .transpose()?;
        let raw = conversation_config.as_ptr();
        unsafe {
            let s = sampler.as_ptr();
            let sampling = config.sampler;
            api.check(
                (api.sampler_params_set_top_k)(s, sampling.top_k),
                "sampler_params_set_top_k",
            )?;
            api.check(
                (api.sampler_params_set_top_p)(s, sampling.top_p),
                "sampler_params_set_top_p",
            )?;
            api.check(
                (api.sampler_params_set_temperature)(s, sampling.temperature),
                "sampler_params_set_temperature",
            )?;
            api.check(
                (api.sampler_params_set_seed)(s, sampling.seed),
                "sampler_params_set_seed",
            )?;
            api.check(
                (api.session_config_set_sampler_params)(session.as_ptr(), s),
                "session_config_set_sampler_params",
            )?;
            if let Some(max_output_tokens) = config.max_output_tokens {
                api.check(
                    (api.session_config_set_max_output_tokens)(
                        session.as_ptr(),
                        c_int_saturating(max_output_tokens),
                    ),
                    "session_config_set_max_output_tokens",
                )?;
            }
            api.check(
                (api.thinking_config_set_enable_thinking)(
                    thinking.as_ptr(),
                    config.enable_thinking,
                ),
                "thinking_config_set_enable_thinking",
            )?;
            api.check(
                (api.conversation_config_set_session_config)(raw, session.as_ptr()),
                "conversation_config_set_session_config",
            )?;
            api.check(
                (api.conversation_config_set_thinking_config)(raw, thinking.as_ptr()),
                "conversation_config_set_thinking_config",
            )?;
            if let Some(system) = &system {
                api.check(
                    (api.conversation_config_set_system_message)(raw, system.as_ptr()),
                    "conversation_config_set_system_message",
                )?;
            }
            if let Some(tools) = &tools {
                api.check(
                    (api.conversation_config_set_tools)(raw, tools.as_ptr()),
                    "conversation_config_set_tools",
                )?;
            }
            if let Some(preface) = &preface {
                api.check(
                    (api.conversation_config_set_messages)(raw, preface.as_ptr()),
                    "conversation_config_set_messages",
                )?;
            }
        }
        let handle = api.create(
            "conversation_create",
            api.conversation_delete,
            |out| unsafe { (api.conversation_create)(self.handle.as_ptr(), raw, out) },
        )?;
        Ok(Conversation {
            handle,
            sink: Box::new(StreamSink {
                api,
                sender: Mutex::new(None),
            }),
            api,
        })
    }
}

/// One streamed chunk, copied out of the callback.
#[derive(Debug)]
pub(super) struct RawChunk {
    pub(super) text: Option<String>,
    pub(super) is_final: bool,
    pub(super) error: Option<String>,
}

/// Benchmark turn counts at one moment.
#[derive(Debug, Clone, Copy)]
pub(super) struct TurnCounts {
    prefill: usize,
    decode: usize,
}

/// The benchmark turns recorded after a [`TurnCounts`] snapshot.
#[derive(Debug, Clone, Default, PartialEq)]
pub(super) struct NewTurns {
    /// Tokens and tokens per second of each prefill turn.
    pub(super) prefill: Vec<(usize, f64)>,
    pub(super) decode_tokens: usize,
}

pub(super) struct Conversation {
    // Declared before `sink`: the conversation (and with it every callback
    // that can still reach the sink) is gone before the sink is freed.
    handle: Owned<RawConversation>,
    sink: Box<StreamSink>,
    api: &'static Api,
}

// SAFETY: as for `Engine`; the streaming callback only touches `StreamSink`,
// which is `Sync`.
unsafe impl Send for Conversation {}

impl Conversation {
    /// Starts a turn. Chunks arrive on the returned channel from LiteRT-LM's
    /// threads; the last one is final or carries an error.
    pub(super) fn send(
        &mut self,
        message_json: &str,
    ) -> Result<mpsc::Receiver<RawChunk>, ProviderError> {
        let message = c_string(message_json, "message")?;
        let (receiver, data) = self.stream();
        let status = unsafe {
            (self.api.conversation_send_message_stream)(
                self.handle.as_ptr(),
                message.as_ptr(),
                ptr::null(),
                ptr::null(),
                stream_callback,
                data,
            )
        };
        self.api.check(status, "conversation_send_message_stream")?;
        Ok(receiver)
    }

    /// Starts a turn from `prompt`, the complete text the template renders for
    /// `messages_json` (every message of the request), after rewinding the KV
    /// cache to `rewind_to_step` tokens. Tokens the cache already holds are
    /// reused rather than prefilled; the messages become the history. Chunks
    /// arrive as for `send`. Needs a giap-main library.
    pub(super) fn send_prompt(
        &mut self,
        rewind_to_step: usize,
        prompt: &str,
        messages_json: &str,
    ) -> Result<mpsc::Receiver<RawChunk>, ProviderError> {
        let giap = self.giap("conversation_send_prefill_text_stream")?;
        let prompt = c_string(prompt, "prompt")?;
        let messages = c_string(messages_json, "messages")?;
        let (receiver, data) = self.stream();
        let status = unsafe {
            (giap.conversation_send_prefill_text_stream)(
                self.handle.as_ptr(),
                c_int_saturating(rewind_to_step),
                prompt.as_ptr(),
                messages.as_ptr(),
                ptr::null(),
                ptr::null(),
                stream_callback,
                data,
            )
        };
        self.api
            .check(status, "conversation_send_prefill_text_stream")?;
        Ok(receiver)
    }

    /// Points the stream sink at a new channel for the next turn.
    fn stream(&mut self) -> (mpsc::Receiver<RawChunk>, *mut c_void) {
        let (sender, receiver) = mpsc::channel();
        *self
            .sink
            .sender
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(sender);
        let data = ptr::from_ref::<StreamSink>(&*self.sink)
            .cast_mut()
            .cast::<c_void>();
        (receiver, data)
    }

    /// The text the template renders for `message_json` as the next message of
    /// this conversation, its preface included while nothing has been sent.
    pub(super) fn render(&self, message_json: &str) -> Result<String, ProviderError> {
        let message = c_string(message_json, "message")?;
        let mut out: *const c_char = ptr::null();
        let status = unsafe {
            (self.api.conversation_render_message_to_string)(
                self.handle.as_ptr(),
                message.as_ptr(),
                &mut out,
            )
        };
        self.api
            .check(status, "conversation_render_message_to_string")?;
        // SAFETY: owned by the conversation until the next render; copied here.
        unsafe { owned_string(out) }.ok_or_else(|| {
            ProviderError::ExecutionError("LiteRT-LM rendered no prompt".to_string())
        })
    }

    /// Writes the KV cache and the tokens it holds to `path` (beside it, then
    /// renamed over it).
    pub(super) fn save_kv_snapshot(&self, path: &Path) -> Result<(), ProviderError> {
        let giap = self.giap("conversation_save_kv_snapshot")?;
        let path = path_c_string(path)?;
        let status =
            unsafe { (giap.conversation_save_kv_snapshot)(self.handle.as_ptr(), path.as_ptr()) };
        self.api.check(status, "conversation_save_kv_snapshot")
    }

    /// Replaces the KV cache with one `save_kv_snapshot` wrote. Follow it with
    /// `send_prompt` from step 0, which reuses the tokens the snapshot holds.
    pub(super) fn load_kv_snapshot(&mut self, path: &Path) -> Result<(), ProviderError> {
        let giap = self.giap("conversation_load_kv_snapshot")?;
        let path = path_c_string(path)?;
        let status =
            unsafe { (giap.conversation_load_kv_snapshot)(self.handle.as_ptr(), path.as_ptr()) };
        self.api.check(status, "conversation_load_kv_snapshot")
    }

    fn giap(&self, call: &str) -> Result<&'static GiapApi, ProviderError> {
        self.api.giap.as_ref().ok_or_else(|| {
            ProviderError::ExecutionError(format!(
                "this LiteRT-LM library has no {call}; build it from GIAP's giap-main"
            ))
        })
    }

    /// After a cancel the conversation must not be used again.
    pub(super) fn cancel(&self) {
        let _ = unsafe { (self.api.conversation_cancel_process)(self.handle.as_ptr()) };
    }

    /// Tokens currently held in the KV cache.
    pub(super) fn token_count(&self) -> Result<usize, ProviderError> {
        let mut count: c_int = 0;
        let status =
            unsafe { (self.api.conversation_get_token_count)(self.handle.as_ptr(), &mut count) };
        self.api.check(status, "conversation_get_token_count")?;
        Ok(usize::try_from(count).unwrap_or(0))
    }

    pub(super) fn turn_counts(&self) -> Option<TurnCounts> {
        let info = self.benchmark()?;
        Some(TurnCounts {
            prefill: info.count(self.api.benchmark_info_get_num_prefill_turns)?,
            decode: info.count(self.api.benchmark_info_get_num_decode_turns)?,
        })
    }

    pub(super) fn turns_since(&self, before: TurnCounts) -> Option<NewTurns> {
        let info = self.benchmark()?;
        let api = self.api;
        let prefill_turns = info.count(api.benchmark_info_get_num_prefill_turns)?;
        let decode_turns = info.count(api.benchmark_info_get_num_decode_turns)?;
        let mut turns = NewTurns::default();
        for index in before.prefill..prefill_turns {
            let tokens = info.at(api.benchmark_info_get_prefill_token_count_at, index)?;
            let mut rate = 0.0f64;
            let status = unsafe {
                (api.benchmark_info_get_prefill_tokens_per_sec_at)(
                    info.handle.as_ptr(),
                    c_int_saturating(index),
                    &mut rate,
                )
            };
            if status != STATUS_OK {
                return None;
            }
            turns.prefill.push((tokens, rate));
        }
        for index in before.decode..decode_turns {
            turns.decode_tokens += info.at(api.benchmark_info_get_decode_token_count_at, index)?;
        }
        Some(turns)
    }

    /// `None` when benchmarking is unavailable.
    fn benchmark(&self) -> Option<BenchmarkInfo> {
        let mut out = ptr::null_mut();
        let status =
            unsafe { (self.api.conversation_get_benchmark_info)(self.handle.as_ptr(), &mut out) };
        if status != STATUS_OK {
            return None;
        }
        NonNull::new(out).map(|ptr| BenchmarkInfo {
            handle: Owned {
                ptr,
                delete: self.api.benchmark_info_delete,
            },
        })
    }
}

struct BenchmarkInfo {
    handle: Owned<RawBenchmarkInfo>,
}

impl BenchmarkInfo {
    fn count(
        &self,
        get: unsafe extern "C" fn(*const RawBenchmarkInfo, *mut c_int) -> Status,
    ) -> Option<usize> {
        let mut count: c_int = 0;
        let status = unsafe { get(self.handle.as_ptr(), &mut count) };
        (status == STATUS_OK).then(|| usize::try_from(count).unwrap_or(0))
    }

    fn at(
        &self,
        get: unsafe extern "C" fn(*const RawBenchmarkInfo, c_int, *mut c_int) -> Status,
        index: usize,
    ) -> Option<usize> {
        let mut value: c_int = 0;
        let status = unsafe { get(self.handle.as_ptr(), c_int_saturating(index), &mut value) };
        (status == STATUS_OK).then(|| usize::try_from(value).unwrap_or(0))
    }
}

/// Where the streaming callback delivers chunks. Lives as long as its
/// conversation so a callback can never outlive it.
struct StreamSink {
    api: &'static Api,
    sender: Mutex<Option<mpsc::Sender<RawChunk>>>,
}

impl StreamSink {
    /// # Safety
    /// `chunk` must be the live chunk handed to the callback.
    unsafe fn read(&self, chunk: *const RawStreamChunk) -> RawChunk {
        let api = self.api;
        let mut text = ptr::null();
        let mut error = ptr::null();
        let mut is_final = false;
        let readable = (api.stream_chunk_get_text)(chunk, &mut text) == STATUS_OK
            && (api.stream_chunk_get_error)(chunk, &mut error) == STATUS_OK
            && (api.stream_chunk_is_final)(chunk, &mut is_final) == STATUS_OK;
        if !readable {
            return RawChunk {
                text: None,
                is_final: true,
                error: Some("LiteRT-LM stream chunk could not be read".to_string()),
            };
        }
        RawChunk {
            text: owned_string(text),
            is_final,
            error: owned_string(error),
        }
    }

    fn deliver(&self, chunk: RawChunk) {
        if let Ok(sender) = self.sender.lock() {
            if let Some(sender) = sender.as_ref() {
                let _ = sender.send(chunk);
            }
        }
    }
}

/// Runs on LiteRT-LM's execution and callback threads, so it copies the chunk
/// and hands it to a channel without blocking.
unsafe extern "C" fn stream_callback(data: *mut c_void, chunk: *const RawStreamChunk) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        if chunk.is_null() {
            return;
        }
        // SAFETY: `data` is the sink of the conversation that started the turn,
        // freed only after that conversation is deleted.
        let Some(sink) = (unsafe { data.cast_const().cast::<StreamSink>().as_ref() }) else {
            return;
        };
        let chunk = unsafe { sink.read(chunk) };
        sink.deliver(chunk);
    }));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_first_directory_holding_the_library_wins() {
        let root = tempfile::tempdir().unwrap();
        let env_dir = root.path().join("env");
        let bundle_dir = root.path().join("bin/litert-lm");
        let exe_dir = root.path().join("bin");
        for dir in [&env_dir, &bundle_dir, &exe_dir] {
            std::fs::create_dir_all(dir).unwrap();
        }
        std::fs::write(bundle_dir.join(LIB_FILE), b"").unwrap();
        std::fs::write(exe_dir.join(LIB_FILE), b"").unwrap();

        let found = locate(&[env_dir.clone(), bundle_dir.clone(), exe_dir]).unwrap();
        assert_eq!(
            found,
            std::fs::canonicalize(bundle_dir.join(LIB_FILE)).unwrap()
        );
        assert!(found.is_absolute());

        std::fs::write(env_dir.join(LIB_FILE), b"").unwrap();
        let found = locate(&[env_dir.clone(), bundle_dir]).unwrap();
        assert_eq!(
            found,
            std::fs::canonicalize(env_dir.join(LIB_FILE)).unwrap()
        );
    }

    #[test]
    fn a_missing_library_names_every_place_searched() {
        let root = tempfile::tempdir().unwrap();
        let error = locate(&[root.path().join("a"), root.path().join("b")]).unwrap_err();
        let ProviderError::ExecutionError(message) = error else {
            panic!("unexpected error kind");
        };
        assert!(message.contains(LIB_FILE));
        assert!(message.contains(&root.path().join("a").display().to_string()));
        assert!(message.contains(&root.path().join("b").display().to_string()));
        assert!(message.contains(LIB_DIR_ENV));
    }

    #[test]
    fn a_broken_library_is_an_error_not_a_crash() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join(LIB_FILE);
        std::fs::write(&path, b"not a shared library").unwrap();
        let Err(ProviderError::ExecutionError(message)) = load_bounded(path) else {
            panic!("loading garbage must fail");
        };
        assert!(message.starts_with("Failed to load LiteRT-LM from"));
    }

    #[test]
    fn status_codes_have_canonical_names() {
        assert_eq!(status_name(3), "INVALID_ARGUMENT");
        assert_eq!(status_name(16), "UNAUTHENTICATED");
        assert_eq!(status_name(2), "UNKNOWN");
        assert_eq!(status_name(99), "UNKNOWN");
    }
}
