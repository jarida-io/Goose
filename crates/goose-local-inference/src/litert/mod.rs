//! LiteRT-LM backend: `.litertlm` models through the LiteRT-LM C API's
//! stateful Conversation. Each loaded model keeps one conversation as its
//! prompt cache; a request that extends it sends only the new messages.
//!
//! With GIAP's giap-main library the cache does more. A request that does not
//! extend the retained conversation (a compacted, edited or repeated history)
//! has its whole prompt prefilled into it from step 0, and LiteRT-LM skips
//! every leading token its KV cache already holds. Before another conversation
//! runs, such as a memory-extraction side call, the retained one is saved to
//! disk as a KV snapshot and deleted, because two live conversations make
//! LiteRT-LM copy the whole KV cache from one into the other; the next request
//! of its family restores the snapshot into a new conversation.

mod convert;
mod fc_repair;
mod ffi;

use std::any::Any;
use std::borrow::Cow;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use goose_provider_types::conversation::message::{Message, MessageContent};
use goose_provider_types::conversation::token_usage::ProviderStats;
use goose_provider_types::errors::ProviderError;
use rmcp::model::{CallToolRequestParams, Role};
use serde_json::Value;
use uuid::Uuid;

use self::convert::{ChunkEvent, ToolCall};
use crate::backend::{BackendLoadedModel, LocalGenerationRequest, LocalInferenceBackend};
use crate::local_model_registry::{ModelSettings, SamplingConfig};
use crate::paths::Paths;
use crate::{finalize_usage, ResolvedModelPaths, StreamSender};

pub(super) const LITERT_BACKEND_ID: &str = "litert";
const DEFAULT_EXECUTION_BACKEND: &str = "cpu";
const DEFAULT_MAX_NUM_TOKENS: usize = 4096;
const DEFAULT_CPU_THREADS: i32 = 4;
const CACHE_SUBDIR: &str = "litert-lm/cache";
const SNAPSHOT_SUBDIR: &str = "litert-lm/kv-snapshots";
/// A conversation holding fewer tokens is cheaper to prefill again than to save.
const MIN_SNAPSHOT_TOKENS: usize = 512;
/// Each snapshot is the whole KV cache, its full window whatever it holds
/// (151 MB for gemma-4-E2B with 16384 tokens, 235 MB for gemma-4-E4B with
/// 8192); the most recently used are kept. Two: one for each chat a household
/// switches between, such as typed and spoken, each with its own system
/// prompt. One-shot work is never saved (`worth_saving`), so it cannot push a
/// chat's out.
const SNAPSHOTS_KEPT: usize = 2;
/// Age at which a snapshot left half-written by a process that died is removed.
const STALE_PARTIAL_SNAPSHOT: Duration = Duration::from_secs(60 * 60);
const STREAM_POLL: Duration = Duration::from_millis(100);

pub(super) struct LiteRtBackend;

impl LiteRtBackend {
    pub(super) fn new() -> Self {
        Self
    }
}

struct LoadedModel {
    // Declared before `engine` so its conversation is deleted first.
    main: Option<MainConversation>,
    /// The conversation last set aside, which stands in for the retained one
    /// while there is none.
    aside: Option<Aside>,
    engine: ffi::Engine,
    max_num_tokens: usize,
    /// `None` when the library lacks giap-main's prompt and snapshot calls.
    snapshots: Option<Snapshots>,
}

/// What a conversation set aside was. While nothing is retained, a request
/// weighing less than half of it is still a side call, so a burst of them
/// after a quiet spell (titling, a memory-extraction pass) neither takes the
/// slot nor writes a snapshot of its own.
struct Aside {
    identity: Identity,
    weight: usize,
}

impl BackendLoadedModel for LoadedModel {
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// The retained conversation and every message it holds, in canonical form,
/// including the turns it generated.
struct MainConversation {
    conversation: ffi::Conversation,
    identity: Identity,
    options: Options,
    consumed: Vec<Value>,
}

impl MainConversation {
    fn held(&self) -> Held<'_> {
        Held {
            identity: &self.identity,
            options: self.options,
            consumed: &self.consumed,
        }
    }

    /// Appends a completed turn: the messages sent and the reply generated.
    fn record(&mut self, sent: &[Value], done: &Completed) {
        self.consumed.extend_from_slice(sent);
        self.consumed
            .extend(convert::generated_message(&done.text, &done.tool_calls));
    }
}

/// What a conversation is fixed to at creation: system message and tools.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Identity {
    system: String,
    tools: Option<String>,
}

/// Conversation-level options that change what the template renders.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Options {
    enable_thinking: bool,
    max_output_tokens: Option<usize>,
}

struct Held<'a> {
    identity: &'a Identity,
    options: Options,
    consumed: &'a [Value],
}

#[derive(Debug, PartialEq, Eq)]
enum Plan {
    /// Send `messages[from..]` to the retained conversation.
    Extend { from: usize },
    /// Prefill the whole prompt into the retained conversation from step 0,
    /// reusing what its KV cache already holds.
    Rematch,
    /// Replace the retained conversation, restoring the request family's KV
    /// snapshot when there is one.
    Recreate,
    /// Answer in a conversation of its own; the retained one is set aside.
    Throwaway,
}

/// How a request uses the retained conversation.
///
/// A different system message or tool set gets its own conversation when it
/// weighs less than half of what is retained, the rule the llama.cpp backend
/// uses for its sacrificial context: side calls such as memory extraction
/// never cost the chat its prefix, while a chat whose tools changed takes the
/// retained slot over instead of running beside it forever. With nothing
/// retained, the conversation last set aside is weighed in its place.
/// `can_rematch` says the library can prefill a whole prompt into the
/// retained conversation.
fn plan(
    held: Option<Held<'_>>,
    aside: Option<&Aside>,
    identity: &Identity,
    options: Options,
    messages: &[Value],
    can_rematch: bool,
) -> Plan {
    let Some(held) = held else {
        return match aside {
            Some(aside)
                if aside.identity != *identity
                    && is_side_call(identity, messages, aside.weight) =>
            {
                Plan::Throwaway
            }
            _ => Plan::Recreate,
        };
    };
    if held.identity != identity {
        return if is_side_call(identity, messages, weight(held.identity, held.consumed)) {
            Plan::Throwaway
        } else {
            Plan::Recreate
        };
    }
    if held.options != options {
        return Plan::Recreate;
    }
    let from = held.consumed.len();
    if messages.len() > from && messages[..from] == *held.consumed {
        Plan::Extend { from }
    } else if can_rematch {
        Plan::Rematch
    } else {
        Plan::Recreate
    }
}

/// A request weighing less than half of the conversation it would displace.
fn is_side_call(identity: &Identity, messages: &[Value], displaced: usize) -> bool {
    weight(identity, messages).saturating_mul(2) < displaced
}

fn weight(identity: &Identity, messages: &[Value]) -> usize {
    identity.system.len()
        + identity.tools.as_ref().map_or(0, String::len)
        + messages
            .iter()
            .map(|message| message.to_string().len())
            .sum::<usize>()
}

/// KV snapshots of conversations set aside, one file per model and family.
struct Snapshots {
    dir: PathBuf,
    /// The model file and the engine settings its KV cache depends on.
    model: String,
}

impl Snapshots {
    fn new(
        model_path: &Path,
        backend: &str,
        max_num_tokens: usize,
        speculative_decoding: bool,
    ) -> Self {
        let (size, modified) = std::fs::metadata(model_path)
            .map(|metadata| {
                let modified = metadata
                    .modified()
                    .ok()
                    .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
                    .map_or(0, |since| since.as_secs());
                (metadata.len(), modified)
            })
            .unwrap_or_default();
        Self {
            dir: Paths::in_data_dir(SNAPSHOT_SUBDIR),
            model: format!(
                "{}\n{size}\n{modified}\n{backend}\n{max_num_tokens}\n{speculative_decoding}",
                model_path.display()
            ),
        }
    }

    /// The family's file: the model, the system message, the tools and whether
    /// thinking is on, everything a conversation is fixed to at creation that
    /// changes what it holds. Keyed on the whole identity, because jobs that
    /// open with the same system prompt as the chat, such as the proactive
    /// review, would otherwise write over the chat's snapshot.
    fn path(&self, identity: &Identity, options: Options) -> PathBuf {
        let hash = fnv1a(&[
            self.model.as_bytes(),
            identity.system.as_bytes(),
            identity.tools.as_deref().unwrap_or_default().as_bytes(),
            &[u8::from(options.enable_thinking)],
        ]);
        self.dir.join(format!("{hash:016x}.kv"))
    }

    fn save(&self, conversation: &ffi::Conversation, path: &Path) -> Result<(), ProviderError> {
        std::fs::create_dir_all(&self.dir).map_err(|error| {
            ProviderError::ExecutionError(format!(
                "Failed to create the LiteRT-LM snapshot directory {}: {error}",
                self.dir.display()
            ))
        })?;
        conversation.save_kv_snapshot(path)?;
        self.prune();
        Ok(())
    }

    /// Keeps the most recently used snapshots and drops abandoned partial ones.
    fn prune(&self) {
        let Ok(entries) = std::fs::read_dir(&self.dir) else {
            return;
        };
        let mut snapshots = Vec::new();
        for entry in entries.flatten() {
            let path = entry.path();
            let Ok(modified) = entry.metadata().and_then(|metadata| metadata.modified()) else {
                continue;
            };
            match path.extension().and_then(|extension| extension.to_str()) {
                Some("kv") => snapshots.push((modified, path)),
                Some("tmp")
                    if modified
                        .elapsed()
                        .is_ok_and(|age| age > STALE_PARTIAL_SNAPSHOT) =>
                {
                    let _ = std::fs::remove_file(path);
                }
                _ => {}
            }
        }
        snapshots.sort_by(|a, b| b.0.cmp(&a.0));
        for (_, path) in snapshots.into_iter().skip(SNAPSHOTS_KEPT) {
            let _ = std::fs::remove_file(path);
        }
    }
}

/// Marks a snapshot as used, for pruning.
fn touch(path: &Path) {
    let _ = std::fs::File::options()
        .write(true)
        .open(path)
        .and_then(|file| file.set_modified(SystemTime::now()));
}

/// FNV-1a over length-prefixed fields; unlike `std`'s hasher it is the same
/// in every build, so snapshot names survive an upgrade.
fn fnv1a(fields: &[&[u8]]) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = OFFSET;
    for field in fields {
        let length = (field.len() as u64).to_le_bytes();
        for byte in length.iter().chain(field.iter()) {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(PRIME);
        }
    }
    hash
}

fn execution_backend(configured: Option<&str>) -> String {
    configured
        .map(|backend| backend.trim().to_ascii_lowercase())
        .filter(|backend| !backend.is_empty())
        .unwrap_or_else(|| DEFAULT_EXECUTION_BACKEND.to_string())
}

/// Always explicit: LiteRT-LM caps an unset value at 4096 on non-Apple GPUs.
fn max_num_tokens(
    context_size: Option<u32>,
    env_limit: Option<usize>,
    model_max: Option<u32>,
) -> usize {
    let requested = context_size
        .filter(|size| *size > 0)
        .map(|size| size as usize)
        .or(env_limit)
        .unwrap_or(DEFAULT_MAX_NUM_TOKENS);
    match model_max {
        Some(max) => requested.min(max as usize),
        None => requested,
    }
}

fn env_context_limit() -> Option<usize> {
    std::env::var("GOOSE_CONTEXT_LIMIT")
        .ok()
        .and_then(|limit| limit.trim().parse::<usize>().ok())
        .filter(|limit| *limit > 0)
}

/// LiteRT-LM's top-k/top-p sampler for every setting; `min_p` has no
/// counterpart, and Mirostat falls back to greedy as in the MLX backend.
fn sampler_config(sampling: &SamplingConfig) -> ffi::SamplerConfig {
    const GREEDY: ffi::SamplerConfig = ffi::SamplerConfig {
        top_k: 1,
        top_p: 1.0,
        temperature: 0.0,
        seed: 0,
    };
    match sampling {
        SamplingConfig::Greedy => GREEDY,
        SamplingConfig::Temperature {
            temperature,
            top_k,
            top_p,
            seed,
            ..
        } => ffi::SamplerConfig {
            top_k: (*top_k).max(1),
            top_p: top_p.clamp(0.0, 1.0),
            temperature: temperature.max(0.0),
            seed: seed.unwrap_or(0) as i32,
        },
        SamplingConfig::MirostatV2 { seed, .. } => ffi::SamplerConfig {
            seed: seed.unwrap_or(0) as i32,
            ..GREEDY
        },
    }
}

fn millis(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

impl LocalInferenceBackend for LiteRtBackend {
    fn id(&self) -> &'static str {
        LITERT_BACKEND_ID
    }

    fn load_model(
        &self,
        model_id: &str,
        resolved: &ResolvedModelPaths,
        settings: &ModelSettings,
    ) -> Result<Box<dyn BackendLoadedModel>, ProviderError> {
        let model_path = &resolved.model_path;
        if !model_path.exists() {
            return Err(ProviderError::ExecutionError(format!(
                "Model not downloaded: {}. Please download it from Settings > Local Inference.",
                model_id
            )));
        }
        let litert = settings.litert.clone().unwrap_or_default();
        let backend = execution_backend(litert.backend.as_deref());

        let file = ffi::LoadedFile::open(model_path)?;
        let max_num_tokens = max_num_tokens(
            settings.context_size,
            env_context_limit(),
            file.max_context_tokens(),
        );
        let speculative_decoding =
            litert.speculative_decoding.unwrap_or(true) && file.supports_speculative_decoding();
        drop(file);

        let cache_dir = Paths::in_data_dir(CACHE_SUBDIR);
        std::fs::create_dir_all(&cache_dir).map_err(|error| {
            ProviderError::ExecutionError(format!(
                "Failed to create the LiteRT-LM cache directory {}: {error}",
                cache_dir.display()
            ))
        })?;
        let num_threads = (backend == "cpu").then(|| {
            settings
                .n_threads
                .filter(|threads| *threads > 0)
                .unwrap_or(DEFAULT_CPU_THREADS)
        });

        tracing::info!(
            backend = LITERT_BACKEND_ID,
            model_id,
            path = %model_path.display(),
            execution = %backend,
            max_num_tokens,
            speculative_decoding,
            "Loading LiteRT-LM model"
        );
        let engine = ffi::Engine::create(&ffi::EngineConfig {
            model_path,
            backend: &backend,
            max_num_tokens,
            cache_dir: &cache_dir,
            num_threads,
            speculative_decoding,
        })?;
        let snapshots = engine
            .supports_snapshots()
            .then(|| Snapshots::new(model_path, &backend, max_num_tokens, speculative_decoding));
        tracing::info!(
            backend = LITERT_BACKEND_ID,
            model_id,
            rematch_and_kv_snapshots = snapshots.is_some(),
            "Loaded LiteRT-LM model"
        );
        Ok(Box::new(LoadedModel {
            main: None,
            aside: None,
            engine,
            max_num_tokens,
            snapshots,
        }))
    }

    fn generate(
        &self,
        loaded: &mut dyn BackendLoadedModel,
        mut request: LocalGenerationRequest<'_>,
    ) -> Result<(), ProviderError> {
        let loaded = loaded
            .as_any_mut()
            .downcast_mut::<LoadedModel>()
            .ok_or_else(|| {
                ProviderError::ExecutionError("Loaded model backend mismatch".to_string())
            })?;
        let started = Instant::now();
        let messages = convert::litert_messages(request.messages);
        if messages.is_empty() {
            return Err(ProviderError::ExecutionError(
                "LiteRT-LM received no messages to answer".to_string(),
            ));
        }
        let identity = Identity {
            system: request.system.to_string(),
            tools: convert::tools_json(request.tools)?,
        };
        let options = Options {
            enable_thinking: request.settings.enable_thinking,
            max_output_tokens: request.settings.max_output_tokens,
        };
        let decision = plan(
            loaded.main.as_ref().map(MainConversation::held),
            loaded.aside.as_ref(),
            &identity,
            options,
            &messages,
            loaded.snapshots.is_some(),
        );
        tracing::debug!(
            target: "giap::kv",
            backend = LITERT_BACKEND_ID,
            plan = ?decision,
            messages = messages.len(),
            "LiteRT-LM conversation plan"
        );

        let turn = match decision {
            Plan::Extend { from } => extend(loaded, &messages, from, &request),
            Plan::Rematch => rematch(loaded, identity, options, &messages, &request),
            Plan::Recreate => {
                set_aside(loaded);
                recreate(loaded, identity, options, &messages, &request, true)
            }
            Plan::Throwaway => throwaway(loaded, &identity, options, &messages, &request),
        };

        if let Turn::Completed(done) = turn? {
            emit_completion(&mut request, done, started, loaded.max_num_tokens);
        }
        Ok(())
    }

    fn available_memory_bytes(&self) -> u64 {
        0
    }
}

fn open_conversation(
    engine: &ffi::Engine,
    identity: &Identity,
    options: Options,
    settings: &ModelSettings,
    preface: &[Value],
) -> Result<ffi::Conversation, ProviderError> {
    let system = convert::system_message_json(&identity.system);
    let preface = (!preface.is_empty()).then(|| Value::Array(preface.to_vec()).to_string());
    engine.conversation(&ffi::ConversationConfig {
        system: system.as_deref(),
        tools: identity.tools.as_deref(),
        preface: preface.as_deref(),
        enable_thinking: options.enable_thinking,
        max_output_tokens: options.max_output_tokens,
        sampler: sampler_config(&settings.sampling),
    })
}

fn extend(
    loaded: &mut LoadedModel,
    messages: &[Value],
    from: usize,
    request: &LocalGenerationRequest<'_>,
) -> Result<Turn, ProviderError> {
    let main = loaded
        .main
        .as_mut()
        .expect("an extend plan has a retained conversation");
    let tail = &messages[from..];
    let turn = run_turn(
        &mut main.conversation,
        &Input::Messages(tail),
        request,
        false,
    );
    match &turn {
        Ok(Turn::Completed(done)) => main.record(tail, done),
        _ => loaded.main = None,
    }
    turn
}

/// Prefills the request's whole prompt into the retained conversation from
/// step 0: LiteRT-LM skips the leading tokens its KV cache already holds, so a
/// compacted, edited or repeated history costs only what changed.
fn rematch(
    loaded: &mut LoadedModel,
    identity: Identity,
    options: Options,
    messages: &[Value],
    request: &LocalGenerationRequest<'_>,
) -> Result<Turn, ProviderError> {
    let input = match prompt_input(
        &loaded.engine,
        &identity,
        options,
        request.settings,
        messages,
    ) {
        Ok(input) => input,
        Err(error) => {
            tracing::warn!(
                backend = LITERT_BACKEND_ID,
                %error,
                "LiteRT-LM could not render the prompt; recreating the conversation"
            );
            loaded.main = None;
            return recreate(loaded, identity, options, messages, request, false);
        }
    };
    let main = loaded
        .main
        .as_mut()
        .expect("a rematch plan has a retained conversation");
    let started = match start_turn(&mut main.conversation, &input, false) {
        Ok(started) => started,
        Err(error) => {
            tracing::warn!(
                backend = LITERT_BACKEND_ID,
                %error,
                "LiteRT-LM could not rematch the retained conversation; recreating it"
            );
            loaded.main = None;
            return recreate(loaded, identity, options, messages, request, false);
        }
    };
    let turn = finish_turn(&mut main.conversation, started, request);
    match &turn {
        Ok(Turn::Completed(done)) => {
            main.consumed.clear();
            main.record(messages, done);
        }
        _ => loaded.main = None,
    }
    turn
}

/// Retains a new conversation for the request.
///
/// With giap-main (and `use_prompt`) the new conversation has no preface:
/// the request's whole prompt is prefilled into it, after the family's KV
/// snapshot, if there is one, is restored. A later rematch then finds the
/// history only where LiteRT-LM keeps it, not repeated in a preface. Otherwise,
/// or when that cannot start, the history is the preface and the last message
/// is sent, as upstream LiteRT-LM expects.
fn recreate(
    loaded: &mut LoadedModel,
    identity: Identity,
    options: Options,
    messages: &[Value],
    request: &LocalGenerationRequest<'_>,
    use_prompt: bool,
) -> Result<Turn, ProviderError> {
    loaded.main = None;
    let history = messages.len() - 1;
    let snapshot = loaded
        .snapshots
        .as_ref()
        .map(|snapshots| snapshots.path(&identity, options))
        .filter(|path| path.is_file());
    if use_prompt && loaded.snapshots.is_some() && (history > 0 || snapshot.is_some()) {
        match prompt_input(
            &loaded.engine,
            &identity,
            options,
            request.settings,
            messages,
        ) {
            Ok(input) => {
                let (mut conversation, restored) = restored_conversation(
                    &loaded.engine,
                    &identity,
                    options,
                    request.settings,
                    snapshot.as_deref(),
                )?;
                match start_turn(&mut conversation, &input, !restored) {
                    Ok(started) => {
                        let turn = finish_turn(&mut conversation, started, request);
                        retain(loaded, conversation, identity, options, messages, &turn);
                        return turn;
                    }
                    Err(error) => tracing::warn!(
                        backend = LITERT_BACKEND_ID,
                        %error,
                        "LiteRT-LM could not prefill the prompt; sending the history as a preface"
                    ),
                }
            }
            Err(error) => tracing::warn!(
                backend = LITERT_BACKEND_ID,
                %error,
                "LiteRT-LM could not render the prompt; sending the history as a preface"
            ),
        }
    }
    let mut conversation = open_conversation(
        &loaded.engine,
        &identity,
        options,
        request.settings,
        &messages[..history],
    )?;
    let turn = run_turn(
        &mut conversation,
        &Input::Messages(&messages[history..]),
        request,
        true,
    );
    retain(loaded, conversation, identity, options, messages, &turn);
    turn
}

/// Keeps a new conversation whose turn completed as the retained one.
fn retain(
    loaded: &mut LoadedModel,
    conversation: ffi::Conversation,
    identity: Identity,
    options: Options,
    messages: &[Value],
    turn: &Result<Turn, ProviderError>,
) {
    if let Ok(Turn::Completed(done)) = turn {
        let mut main = MainConversation {
            conversation,
            identity,
            options,
            consumed: Vec::new(),
        };
        main.record(messages, done);
        loaded.main = Some(main);
    }
}

/// Answers a side call, such as memory extraction, in a conversation of its
/// own. With giap-main the retained conversation is set aside first; without
/// it, it stays, and LiteRT-LM copies its KV cache aside while the call runs.
fn throwaway(
    loaded: &mut LoadedModel,
    identity: &Identity,
    options: Options,
    messages: &[Value],
    request: &LocalGenerationRequest<'_>,
) -> Result<Turn, ProviderError> {
    if loaded.snapshots.is_some() {
        set_aside(loaded);
    }
    let history = messages.len() - 1;
    let mut conversation = open_conversation(
        &loaded.engine,
        identity,
        options,
        request.settings,
        &messages[..history],
    )?;
    run_turn(
        &mut conversation,
        &Input::Messages(&messages[history..]),
        request,
        true,
    )
}

/// Deletes the retained conversation, first saving its KV cache as its
/// family's snapshot when the library can and it holds enough to be worth
/// restoring.
fn set_aside(loaded: &mut LoadedModel) {
    let Some(main) = loaded.main.take() else {
        return;
    };
    let Some(snapshots) = &loaded.snapshots else {
        return;
    };
    loaded.aside = Some(Aside {
        identity: main.identity.clone(),
        weight: weight(&main.identity, &main.consumed),
    });
    let tokens = main.conversation.token_count().unwrap_or(0);
    if !worth_saving(&main.identity, tokens) {
        return;
    }
    let path = snapshots.path(&main.identity, main.options);
    let saving = Instant::now();
    match snapshots.save(&main.conversation, &path) {
        Ok(()) => tracing::info!(
            target: "giap::kv",
            backend = LITERT_BACKEND_ID,
            tokens,
            save_ms = millis(saving.elapsed()),
            file = %path.display(),
            "Saved the retained LiteRT-LM conversation as a KV snapshot"
        ),
        Err(error) => tracing::warn!(
            backend = LITERT_BACKEND_ID,
            %error,
            "Could not save the retained LiteRT-LM conversation's KV snapshot"
        ),
    }
}

/// Whether a conversation set aside will be asked again, so its KV cache is
/// worth a snapshot. A chat carries tools; one-shot work, such as compacting a
/// conversation, summarising one or naming one, carries none and is never
/// resumed. On a Jetson Orin every in-turn compaction saved its own
/// conversation too (5,261 tokens, 235 MB, 1.5 s) while memory was tightest,
/// and it took one of the kept files from a chat.
fn worth_saving(identity: &Identity, tokens: usize) -> bool {
    identity.tools.is_some() && tokens >= MIN_SNAPSHOT_TOKENS
}

/// A new conversation without a preface, holding the snapshot at `snapshot`
/// when that loads; says whether it does. A snapshot that does not load is
/// deleted.
fn restored_conversation(
    engine: &ffi::Engine,
    identity: &Identity,
    options: Options,
    settings: &ModelSettings,
    snapshot: Option<&Path>,
) -> Result<(ffi::Conversation, bool), ProviderError> {
    let mut conversation = open_conversation(engine, identity, options, settings, &[])?;
    let Some(path) = snapshot else {
        return Ok((conversation, false));
    };
    let loading = Instant::now();
    match conversation.load_kv_snapshot(path) {
        Ok(()) => {
            touch(path);
            tracing::info!(
                target: "giap::kv",
                backend = LITERT_BACKEND_ID,
                tokens = conversation.token_count().unwrap_or(0),
                load_ms = millis(loading.elapsed()),
                file = %path.display(),
                "Restored a LiteRT-LM KV snapshot"
            );
            Ok((conversation, true))
        }
        Err(error) => {
            tracing::warn!(
                backend = LITERT_BACKEND_ID,
                %error,
                file = %path.display(),
                "Deleting a LiteRT-LM KV snapshot that did not load"
            );
            let _ = std::fs::remove_file(path);
            // What a failed load left in the KV cache is not trusted.
            drop(conversation);
            let conversation = open_conversation(engine, identity, options, settings, &[])?;
            Ok((conversation, false))
        }
    }
}

/// The request's whole prompt as a new conversation would prefill it,
/// rendered by a conversation that is never run, so no KV cache moves.
fn prompt_input(
    engine: &ffi::Engine,
    identity: &Identity,
    options: Options,
    settings: &ModelSettings,
    messages: &[Value],
) -> Result<Input<'static>, ProviderError> {
    let (last, preface) = messages.split_last().ok_or_else(|| {
        ProviderError::ExecutionError("LiteRT-LM received no messages to answer".to_string())
    })?;
    let renderer = open_conversation(engine, identity, options, settings, preface)?;
    let text = renderer.render(&last.to_string())?;
    Ok(Input::Prompt {
        text,
        messages: convert::send_payload(messages),
    })
}

/// What a turn sends.
enum Input<'a> {
    /// The messages after those the conversation holds.
    Messages(&'a [Value]),
    /// A request's whole prompt, prefilled from step 0 so LiteRT-LM reuses
    /// the leading tokens its KV cache holds; `messages`, every message of the
    /// request, become the conversation's history.
    Prompt { text: String, messages: String },
}

enum Turn {
    Completed(Completed),
    /// goose dropped the stream; the conversation was cancelled.
    Abandoned,
}

struct Completed {
    text: String,
    thinking: String,
    tool_calls: Vec<ToolCall>,
    time_to_first_token_ms: Option<u64>,
    numbers: TurnNumbers,
}

/// KV and benchmark readings around one turn.
#[derive(Debug, Clone, Default, PartialEq)]
struct TurnNumbers {
    tokens_before: usize,
    tokens_after: Option<usize>,
    new_turns: Option<ffi::NewTurns>,
    /// The whole prompt was prefilled from step 0. LiteRT-LM's benchmark then
    /// counts every prompt token as prefilled, the ones it skipped included,
    /// so how many were reused is not known here (its log line says).
    whole_prompt: bool,
}

impl TurnNumbers {
    /// Time spent in this turn's prefills, from LiteRT-LM's per-turn rates.
    fn prefill_ms(&self) -> Option<u64> {
        let turns = &self.new_turns.as_ref()?.prefill;
        if turns.is_empty() {
            return None;
        }
        let ms: f64 = turns
            .iter()
            .filter(|(_, rate)| *rate > 0.0)
            .map(|(tokens, rate)| *tokens as f64 / rate * 1000.0)
            .sum();
        Some(ms.round() as u64)
    }

    fn decode_tokens(&self) -> Option<usize> {
        self.new_turns.as_ref().map(|turns| turns.decode_tokens)
    }

    /// Tokens already in the KV cache that this turn did not prefill again.
    /// Lower than the count before the send when LiteRT-LM rewound to drop
    /// earlier thoughts and refilled from there.
    fn reused_prefix_tokens(&self) -> Option<usize> {
        if self.whole_prompt {
            return None;
        }
        Some(match (self.tokens_after, &self.new_turns) {
            (Some(after), Some(turns)) => {
                let prefilled: usize = turns.prefill.iter().map(|(tokens, _)| tokens).sum();
                after
                    .saturating_sub(prefilled + turns.decode_tokens)
                    .min(self.tokens_before)
            }
            _ => self.tokens_before,
        })
    }

    /// Context the model attended to before generating.
    fn prompt_tokens(&self) -> usize {
        match (self.tokens_after, self.decode_tokens()) {
            (Some(after), Some(decoded)) => after.saturating_sub(decoded),
            (Some(after), None) => after,
            (None, _) => self.tokens_before,
        }
    }
}

/// A turn whose stream has started.
struct Started {
    chunks: Receiver<ffi::RawChunk>,
    sent: Instant,
    tokens_before: usize,
    turns_before: Option<ffi::TurnCounts>,
    whole_prompt: bool,
}

fn run_turn(
    conversation: &mut ffi::Conversation,
    input: &Input<'_>,
    request: &LocalGenerationRequest<'_>,
    fresh: bool,
) -> Result<Turn, ProviderError> {
    let started = start_turn(conversation, input, fresh)?;
    finish_turn(conversation, started, request)
}

/// Sends the input. An error here means nothing was generated, so the caller
/// can still try another way.
fn start_turn(
    conversation: &mut ffi::Conversation,
    input: &Input<'_>,
    fresh: bool,
) -> Result<Started, ProviderError> {
    let tokens_before = if fresh {
        0
    } else {
        conversation.token_count().unwrap_or(0)
    };
    let turns_before = conversation.turn_counts();
    let sent = Instant::now();
    let chunks = match input {
        Input::Messages(tail) => conversation.send(&convert::send_payload(tail))?,
        Input::Prompt { text, messages } => conversation.send_prompt(0, text, messages)?,
    };
    Ok(Started {
        chunks,
        sent,
        tokens_before,
        turns_before,
        whole_prompt: matches!(input, Input::Prompt { .. }),
    })
}

fn finish_turn(
    conversation: &mut ffi::Conversation,
    started: Started,
    request: &LocalGenerationRequest<'_>,
) -> Result<Turn, ProviderError> {
    let mut stream = TurnStream::new(request.message_id, request.tx, started.sent);
    loop {
        let chunk = match started.chunks.recv_timeout(STREAM_POLL) {
            Ok(chunk) => chunk,
            Err(RecvTimeoutError::Timeout) => {
                if request.tx.is_closed() {
                    conversation.cancel();
                    return Ok(Turn::Abandoned);
                }
                continue;
            }
            Err(RecvTimeoutError::Disconnected) => {
                return Err(ProviderError::ExecutionError(
                    "LiteRT-LM stream ended without a final chunk".to_string(),
                ));
            }
        };
        for event in convert::chunk_events(
            chunk.text.as_deref(),
            chunk.is_final,
            chunk.error.as_deref(),
        ) {
            let delivered = match event {
                ChunkEvent::TextDelta(text) => stream.text(text),
                ChunkEvent::ThoughtDelta(thought) => stream.thought(thought),
                ChunkEvent::ToolCalls(calls) => {
                    stream.tool_calls.extend(calls);
                    true
                }
                ChunkEvent::Error(message) => {
                    conversation.cancel();
                    return Err(convert::stream_error(&message));
                }
                ChunkEvent::Final => {
                    let numbers = TurnNumbers {
                        tokens_before: started.tokens_before,
                        tokens_after: conversation.token_count().ok(),
                        new_turns: started
                            .turns_before
                            .and_then(|before| conversation.turns_since(before)),
                        whole_prompt: started.whole_prompt,
                    };
                    return Ok(Turn::Completed(stream.finish(numbers)));
                }
            };
            if !delivered {
                conversation.cancel();
                return Ok(Turn::Abandoned);
            }
        }
    }
}

/// Streams deltas in the shapes the llama.cpp native-tools path emits: text
/// and thinking as messages sharing one id.
struct TurnStream<'a> {
    message_id: &'a str,
    tx: &'a StreamSender,
    sent: Instant,
    first_piece: Option<Instant>,
    text: String,
    thinking: String,
    tool_calls: Vec<ToolCall>,
}

impl<'a> TurnStream<'a> {
    fn new(message_id: &'a str, tx: &'a StreamSender, sent: Instant) -> Self {
        Self {
            message_id,
            tx,
            sent,
            first_piece: None,
            text: String::new(),
            thinking: String::new(),
            tool_calls: Vec::new(),
        }
    }

    fn text(&mut self, text: String) -> bool {
        self.first_piece.get_or_insert_with(Instant::now);
        self.text.push_str(&text);
        self.deliver(Message::assistant().with_text(text))
    }

    fn thought(&mut self, thought: String) -> bool {
        self.first_piece.get_or_insert_with(Instant::now);
        self.thinking.push_str(&thought);
        self.deliver(Message::assistant().with_thinking(thought, ""))
    }

    fn deliver(&self, mut message: Message) -> bool {
        message.id = Some(self.message_id.to_string());
        self.tx.blocking_send(Ok((Some(message), None))).is_ok()
    }

    fn finish(self, numbers: TurnNumbers) -> Completed {
        Completed {
            time_to_first_token_ms: self
                .first_piece
                .map(|first| millis(first.duration_since(self.sent))),
            text: self.text,
            thinking: self.thinking,
            tool_calls: self.tool_calls,
            numbers,
        }
    }
}

/// Sends the turn's tool calls as one message, then the usage.
fn emit_completion(
    request: &mut LocalGenerationRequest<'_>,
    done: Completed,
    started: Instant,
    max_num_tokens: usize,
) {
    if !done.tool_calls.is_empty() {
        let round = Uuid::new_v4().as_u64_pair().0;
        let mut contents = Vec::with_capacity(done.tool_calls.len() + 1);
        if !done.thinking.is_empty() {
            contents.push(MessageContent::thinking(&done.thinking, ""));
        }
        contents.extend(done.tool_calls.iter().enumerate().map(|(index, call)| {
            MessageContent::tool_request(
                convert::tool_call_id(round, index),
                Ok(CallToolRequestParams::new(Cow::Owned(call.name.clone()))
                    .with_arguments(call.arguments.clone())),
            )
        }));
        let mut message = Message::new(Role::Assistant, chrono::Utc::now().timestamp(), contents);
        message.id = Some(request.message_id.to_string());
        let _ = request.tx.blocking_send(Ok((Some(message), None)));
    }

    let numbers = &done.numbers;
    let output_tokens = numbers.decode_tokens();
    let stats = ProviderStats {
        time_to_first_token_ms: done.time_to_first_token_ms,
        model_load_ms: request.model_load_ms,
        elapsed_ms: Some(millis(started.elapsed())),
        output_tokens,
        draft: None,
        prefill_ms: numbers.prefill_ms(),
        effective_context_tokens: Some(max_num_tokens),
        reused_prefix_tokens: numbers.reused_prefix_tokens(),
    };
    let usage = finalize_usage(
        &mut *request.log,
        std::mem::take(&mut request.model_name),
        LITERT_BACKEND_ID,
        numbers.prompt_tokens(),
        i32::try_from(output_tokens.unwrap_or(0)).unwrap_or(i32::MAX),
        Some(stats),
        Some(("generated_text", &done.text)),
    );
    let _ = request.tx.blocking_send(Ok((None, Some(usage))));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::local_model_registry::LiteRtSettings;
    use goose_provider_types::conversation::token_usage::ProviderUsage;
    use rmcp::model::{CallToolResult, Content, Tool};
    use serde_json::json;
    use std::path::PathBuf;

    fn identity(system: &str) -> Identity {
        Identity {
            system: system.to_string(),
            tools: Some(r#"[{"type":"function"}]"#.to_string()),
        }
    }

    const OPTIONS: Options = Options {
        enable_thinking: true,
        max_output_tokens: None,
    };

    fn user(text: &str) -> Value {
        json!({"role": "user", "content": [{"type": "text", "text": text}]})
    }

    fn assistant(text: &str) -> Value {
        json!({"role": "assistant", "content": [{"type": "text", "text": text}]})
    }

    fn held<'a>(identity: &'a Identity, consumed: &'a [Value]) -> Option<Held<'a>> {
        Some(Held {
            identity,
            options: OPTIONS,
            consumed,
        })
    }

    #[test]
    fn nothing_retained_creates_the_conversation() {
        for can_rematch in [false, true] {
            assert_eq!(
                plan(
                    None,
                    None,
                    &identity("chat"),
                    OPTIONS,
                    &[user("hi")],
                    can_rematch
                ),
                Plan::Recreate
            );
        }
    }

    #[test]
    fn a_request_extending_the_conversation_sends_only_its_tail() {
        let chat = identity("chat");
        let consumed = vec![user("hi"), assistant("hello")];
        let request = vec![user("hi"), assistant("hello"), user("weather?")];
        for can_rematch in [false, true] {
            assert_eq!(
                plan(
                    held(&chat, &consumed),
                    None,
                    &chat,
                    OPTIONS,
                    &request,
                    can_rematch
                ),
                Plan::Extend { from: 2 }
            );
        }
    }

    #[test]
    fn a_diverged_or_repeated_history_rematches_when_the_library_can() {
        let chat = identity("chat");
        let consumed = vec![user("hi"), assistant("hello")];
        let edited = vec![user("hey"), assistant("hello"), user("weather?")];
        let compacted = vec![user("summary"), user("weather?")];
        for request in [&edited, &consumed, &compacted] {
            assert_eq!(
                plan(held(&chat, &consumed), None, &chat, OPTIONS, request, true),
                Plan::Rematch
            );
            assert_eq!(
                plan(held(&chat, &consumed), None, &chat, OPTIONS, request, false),
                Plan::Recreate
            );
        }
    }

    #[test]
    fn changed_conversation_options_recreate() {
        let chat = identity("chat");
        let consumed = vec![user("hi"), assistant("hello")];
        let request = vec![user("hi"), assistant("hello"), user("weather?")];
        let no_thinking = Options {
            enable_thinking: false,
            ..OPTIONS
        };
        for can_rematch in [false, true] {
            assert_eq!(
                plan(
                    held(&chat, &consumed),
                    None,
                    &chat,
                    no_thinking,
                    &request,
                    can_rematch
                ),
                Plan::Recreate
            );
        }
    }

    #[test]
    fn a_small_side_call_gets_a_conversation_of_its_own() {
        let chat = Identity {
            system: "chat ".repeat(500),
            tools: Some("tool ".repeat(500)),
        };
        let consumed = vec![user("hi"), assistant("hello")];
        let extraction = Identity {
            system: "Extract memories.".to_string(),
            tools: None,
        };
        for can_rematch in [false, true] {
            assert_eq!(
                plan(
                    held(&chat, &consumed),
                    None,
                    &extraction,
                    OPTIONS,
                    &[user("the user said hi")],
                    can_rematch
                ),
                Plan::Throwaway
            );
        }
    }

    #[test]
    fn a_comparable_new_identity_takes_the_retained_slot() {
        let chat = identity(&"chat ".repeat(100));
        let consumed = vec![user("hi"), assistant("hello")];
        let retooled = Identity {
            system: chat.system.clone(),
            tools: Some(r#"[{"type":"function"},{"type":"function"}]"#.to_string()),
        };
        let request = vec![user("hi"), assistant("hello"), user("weather?")];
        for can_rematch in [false, true] {
            assert_eq!(
                plan(
                    held(&chat, &consumed),
                    None,
                    &retooled,
                    OPTIONS,
                    &request,
                    can_rematch
                ),
                Plan::Recreate
            );
        }
    }

    #[test]
    fn a_side_call_after_the_chat_was_set_aside_does_not_take_the_slot() {
        let chat = Identity {
            system: "chat ".repeat(500),
            tools: Some("tool ".repeat(500)),
        };
        let aside = Aside {
            weight: weight(&chat, &[user("hi"), assistant("hello")]),
            identity: chat.clone(),
        };
        let titling = Identity {
            system: "Title this conversation.".to_string(),
            tools: None,
        };
        assert_eq!(
            plan(None, Some(&aside), &titling, OPTIONS, &[user("hi")], true),
            Plan::Throwaway
        );
        // The chat comes back, and so does a request that weighs as much.
        let request = vec![user("hi"), assistant("hello"), user("weather?")];
        assert_eq!(
            plan(None, Some(&aside), &chat, OPTIONS, &request, true),
            Plan::Recreate
        );
        let review = Identity {
            system: "review ".repeat(400),
            tools: Some("tool ".repeat(400)),
        };
        assert_eq!(
            plan(
                None,
                Some(&aside),
                &review,
                OPTIONS,
                &[user("review the day")],
                true
            ),
            Plan::Recreate
        );
    }

    fn snapshot_store(dir: &Path) -> Snapshots {
        Snapshots {
            dir: dir.to_path_buf(),
            model: "/models/gemma.litertlm\n100\n200\ngpu\n8192\nfalse".to_string(),
        }
    }

    #[test]
    fn a_snapshot_family_is_everything_a_conversation_is_fixed_to() {
        let dir = PathBuf::from("/snapshots");
        let store = snapshot_store(&dir);
        let prompt = "You are the household assistant. ".repeat(20);
        let chat = Identity {
            system: prompt.clone(),
            tools: Some("[1]".to_string()),
        };
        let path = store.path(&chat, OPTIONS);
        assert_eq!(path.parent(), Some(dir.as_path()));
        assert_eq!(path.extension(), Some("kv".as_ref()));
        // The output limit does not change what the KV cache holds.
        let bounded = Options {
            max_output_tokens: Some(64),
            ..OPTIONS
        };
        assert_eq!(store.path(&chat, bounded), path);

        // A job that opens like the chat, such as the proactive review, never
        // shares its file.
        let review = Identity {
            system: format!("{prompt}Review the household's day."),
            tools: Some("[1]".to_string()),
        };
        assert_ne!(store.path(&review, OPTIONS), path);
        let retooled = Identity {
            tools: Some("[1,2]".to_string()),
            ..chat.clone()
        };
        assert_ne!(store.path(&retooled, OPTIONS), path);
        let toolless = Identity {
            tools: None,
            ..chat.clone()
        };
        assert_ne!(store.path(&toolless, OPTIONS), path);
        let no_thinking = Options {
            enable_thinking: false,
            ..OPTIONS
        };
        assert_ne!(store.path(&chat, no_thinking), path);
        let other_model = Snapshots {
            model: "/models/other.litertlm".to_string(),
            ..snapshot_store(&dir)
        };
        assert_ne!(other_model.path(&chat, OPTIONS), path);
    }

    #[test]
    fn pruning_keeps_the_newest_snapshots_and_drops_abandoned_partial_ones() {
        let dir = tempfile::tempdir().unwrap();
        let now = SystemTime::now();
        let write = |name: &str, age: Duration| {
            let path = dir.path().join(name);
            std::fs::write(&path, b"kv").unwrap();
            std::fs::File::options()
                .write(true)
                .open(&path)
                .unwrap()
                .set_modified(now - age)
                .unwrap();
        };
        for kept in 0..SNAPSHOTS_KEPT {
            write(
                &format!("kept-{kept}.kv"),
                Duration::from_secs(10 + kept as u64),
            );
        }
        write("oldest.kv", Duration::from_secs(60));
        write("abandoned.kv.tmp", STALE_PARTIAL_SNAPSHOT * 2);
        write("writing.kv.tmp", Duration::from_secs(5));

        snapshot_store(dir.path()).prune();
        let mut left: Vec<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .collect();
        left.sort();
        let mut expected: Vec<String> = (0..SNAPSHOTS_KEPT)
            .map(|kept| format!("kept-{kept}.kv"))
            .chain(["writing.kv.tmp".to_string()])
            .collect();
        expected.sort();
        assert_eq!(left, expected);
    }

    #[test]
    fn only_a_chat_holding_enough_is_worth_a_snapshot() {
        let chat = Identity {
            system: "You are the household assistant.".to_string(),
            tools: Some("[{\"name\":\"get_weather\"}]".to_string()),
        };
        let compaction = Identity {
            system: "Distill the conversation below into a summary.".to_string(),
            tools: None,
        };
        assert!(worth_saving(&chat, 6_580));
        assert!(!worth_saving(&chat, MIN_SNAPSHOT_TOKENS - 1));
        assert!(
            !worth_saving(&compaction, 5_261),
            "a compaction call is never resumed, however long it was"
        );
    }

    #[test]
    fn snapshot_names_are_stable_and_fields_stay_apart() {
        // FNV-1a 64 over the little-endian length and the bytes, computed
        // independently; a change here orphans every saved snapshot.
        assert_eq!(fnv1a(&[]), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a(&[b"chat"]), 0x508a_ff55_cbf4_5cff);
        assert_ne!(fnv1a(&[b"ab", b"c"]), fnv1a(&[b"a", b"bc"]));
    }

    #[test]
    fn the_token_budget_is_always_explicit() {
        assert_eq!(max_num_tokens(Some(16384), Some(8192), None), 16384);
        assert_eq!(max_num_tokens(None, Some(8192), None), 8192);
        assert_eq!(max_num_tokens(Some(0), None, None), DEFAULT_MAX_NUM_TOKENS);
        assert_eq!(max_num_tokens(None, None, None), DEFAULT_MAX_NUM_TOKENS);
        assert_eq!(max_num_tokens(Some(65536), None, Some(32768)), 32768);
    }

    #[test]
    fn execution_backend_names_are_normalised() {
        assert_eq!(execution_backend(Some(" GPU ")), "gpu");
        assert_eq!(execution_backend(Some("")), DEFAULT_EXECUTION_BACKEND);
        assert_eq!(execution_backend(None), DEFAULT_EXECUTION_BACKEND);
    }

    #[test]
    fn every_sampling_setting_maps_to_top_k_top_p() {
        let greedy = sampler_config(&SamplingConfig::Greedy);
        assert_eq!((greedy.top_k, greedy.temperature), (1, 0.0));
        assert_eq!(
            sampler_config(&SamplingConfig::Temperature {
                temperature: 0.8,
                top_k: 40,
                top_p: 0.95,
                min_p: 0.05,
                seed: Some(7),
            }),
            ffi::SamplerConfig {
                top_k: 40,
                top_p: 0.95,
                temperature: 0.8,
                seed: 7,
            }
        );
        let unbounded = sampler_config(&SamplingConfig::Temperature {
            temperature: -1.0,
            top_k: 0,
            top_p: 1.5,
            min_p: 0.0,
            seed: None,
        });
        assert_eq!(
            (unbounded.top_k, unbounded.top_p, unbounded.temperature),
            (1, 1.0, 0.0)
        );
        let mirostat = sampler_config(&SamplingConfig::MirostatV2 {
            tau: 5.0,
            eta: 0.1,
            seed: Some(3),
        });
        assert_eq!((mirostat.top_k, mirostat.seed), (1, 3));
    }

    #[test]
    fn turn_numbers_derive_honest_stats() {
        let extended = TurnNumbers {
            tokens_before: 1000,
            tokens_after: Some(1250),
            new_turns: Some(ffi::NewTurns {
                prefill: vec![(200, 1000.0)],
                decode_tokens: 50,
            }),
            whole_prompt: false,
        };
        assert_eq!(extended.prefill_ms(), Some(200));
        assert_eq!(extended.reused_prefix_tokens(), Some(1000));
        assert_eq!(extended.prompt_tokens(), 1200);
        assert_eq!(extended.decode_tokens(), Some(50));

        let rewound = TurnNumbers {
            tokens_before: 1000,
            tokens_after: Some(1250),
            new_turns: Some(ffi::NewTurns {
                prefill: vec![(600, 1200.0), (200, 400.0)],
                decode_tokens: 50,
            }),
            whole_prompt: false,
        };
        assert_eq!(rewound.reused_prefix_tokens(), Some(400));
        assert_eq!(rewound.prefill_ms(), Some(1000));

        let unmeasured = TurnNumbers {
            tokens_before: 300,
            tokens_after: None,
            new_turns: None,
            whole_prompt: false,
        };
        assert_eq!(unmeasured.prefill_ms(), None);
        assert_eq!(unmeasured.reused_prefix_tokens(), Some(300));
        assert_eq!(unmeasured.prompt_tokens(), 300);

        // The benchmark counts the skipped tokens of a whole prompt as
        // prefilled; its time is still the real prefill time.
        let restored = TurnNumbers {
            tokens_before: 6000,
            tokens_after: Some(6250),
            new_turns: Some(ffi::NewTurns {
                prefill: vec![(6200, 31000.0)],
                decode_tokens: 50,
            }),
            whole_prompt: true,
        };
        assert_eq!(restored.reused_prefix_tokens(), None);
        assert_eq!(restored.prefill_ms(), Some(200));
        assert_eq!(restored.prompt_tokens(), 6200);
    }

    struct Reply {
        text: String,
        tool_requests: Vec<(String, String, serde_json::Map<String, Value>)>,
        stats: Option<ProviderStats>,
        input_tokens: Option<i32>,
        error: Option<ProviderError>,
    }

    fn one_parameter_tool(name: &str, description: &str, parameter: &str, about: &str) -> Tool {
        Tool::new(
            name.to_string(),
            description.to_string(),
            std::sync::Arc::new(
                json!({
                    "type": "object",
                    "properties": {parameter: {"type": "string", "description": about}},
                    "required": [parameter],
                })
                .as_object()
                .cloned()
                .unwrap(),
            ),
        )
    }

    fn weather_tool() -> Tool {
        one_parameter_tool(
            "weather__get_weather",
            "Get the current weather for a city.",
            "city",
            "City name",
        )
    }

    /// A household's tool list, long enough that the chat is worth a KV
    /// snapshot when a side call sets it aside.
    fn household_tools() -> Vec<Tool> {
        let mut tools = vec![weather_tool()];
        for (name, description, parameter, about) in [
            (
                "lights__set_brightness",
                "Set the brightness of the lights in a room.",
                "room",
                "Room name, such as kitchen",
            ),
            (
                "timer__start",
                "Start a countdown timer.",
                "duration",
                "Duration such as 10 minutes",
            ),
            (
                "music__play",
                "Play a song, artist, album or playlist.",
                "query",
                "What to play",
            ),
            (
                "calendar__list_events",
                "List the household's calendar events for a day.",
                "day",
                "Day such as today or 2026-10-05",
            ),
            (
                "reminders__add",
                "Add a reminder for a member of the household.",
                "text",
                "What to be reminded of",
            ),
            (
                "news__headlines",
                "Read the latest news headlines on a topic.",
                "topic",
                "Topic such as sport or Kenya",
            ),
            (
                "wikipedia__search",
                "Look up a topic on Wikipedia.",
                "query",
                "Topic to look up",
            ),
            (
                "shopping__add_item",
                "Add an item to the shared shopping list.",
                "item",
                "Item such as milk",
            ),
            (
                "thermostat__set",
                "Set the target temperature of the heating.",
                "celsius",
                "Temperature in degrees Celsius",
            ),
            (
                "translate__text",
                "Translate text into another language.",
                "text",
                "Text to translate",
            ),
            (
                "recipes__find",
                "Find a recipe from the ingredients at hand.",
                "ingredients",
                "Comma-separated ingredients",
            ),
        ] {
            tools.push(one_parameter_tool(name, description, parameter, about));
        }
        tools
    }

    fn generate_once(
        backend: &LiteRtBackend,
        loaded: &mut dyn BackendLoadedModel,
        resolved: &ResolvedModelPaths,
        system: &str,
        messages: &[Message],
        tools: &[Tool],
        stop_after_first_piece: bool,
    ) -> Reply {
        let (tx, mut rx) = tokio::sync::mpsc::channel(32);
        let mut reply = Reply {
            text: String::new(),
            tool_requests: Vec::new(),
            stats: None,
            input_tokens: None,
            error: None,
        };
        let outcome = std::thread::scope(|scope| {
            let worker = scope.spawn(move || {
                let mut log = None;
                let message_id = Uuid::new_v4().to_string();
                backend.generate(
                    loaded,
                    LocalGenerationRequest {
                        model_name: "gemma-4-E2B-it".to_string(),
                        system,
                        messages,
                        tools,
                        settings: &resolved.settings,
                        temperature: None,
                        max_tokens: None,
                        context_limit: resolved.context_limit,
                        model_load_ms: None,
                        resolved_model: resolved,
                        draft_model_path: None,
                        message_id: &message_id,
                        tx: &tx,
                        log: &mut log,
                    },
                )
            });
            while let Some(item) = rx.blocking_recv() {
                match item {
                    Ok((Some(message), usage)) => {
                        for content in &message.content {
                            match content {
                                MessageContent::Text(text) => reply.text.push_str(&text.text),
                                MessageContent::ToolRequest(request) => {
                                    let call = request.tool_call.as_ref().unwrap();
                                    reply.tool_requests.push((
                                        request.id.clone(),
                                        call.name.to_string(),
                                        call.arguments.clone().unwrap_or_default(),
                                    ));
                                }
                                _ => {}
                            }
                        }
                        if let Some(usage) = usage {
                            reply.input_tokens = usage.usage.input_tokens;
                            reply.stats = usage.stats;
                        }
                        if stop_after_first_piece {
                            break;
                        }
                    }
                    Ok((None, Some(ProviderUsage { usage, stats, .. }))) => {
                        reply.input_tokens = usage.input_tokens;
                        reply.stats = stats;
                    }
                    Ok((None, None)) => {}
                    Err(error) => reply.error = Some(error),
                }
            }
            drop(rx);
            worker.join().unwrap()
        });
        if let Err(error) = outcome {
            reply.error = Some(error);
        }
        reply
    }

    /// A real conversation with a tool round, a side call, a regenerated
    /// turn and a cancelled one. Run with
    /// `cargo test -p goose-local-inference -- --ignored litert`,
    /// `GOOSE_LITERT_LIB_DIR` and `GOOSE_LITERT_TEST_MODEL` set.
    /// `GOOSE_LITERT_TEST_BACKEND` picks `gpu` (default) or `cpu`;
    /// `GOOSE_LITERT_TEST_THINKING` turns thinking on. With a giap-main
    /// library it also checks that the side call sets the chat aside as a KV
    /// snapshot and the next turn restores it.
    #[test]
    #[ignore = "needs the LiteRT-LM library and a .litertlm model"]
    fn litert_live_conversation_with_a_tool() {
        let (Some(_), Some(model)) = (
            std::env::var_os(ffi::LIB_DIR_ENV),
            std::env::var_os("GOOSE_LITERT_TEST_MODEL"),
        ) else {
            eprintln!(
                "skipped: set {} and GOOSE_LITERT_TEST_MODEL",
                ffi::LIB_DIR_ENV
            );
            return;
        };
        let root = std::env::temp_dir().join("goose-litert-live-test");
        let _env = env_lock::lock_env([("GOOSE_PATH_ROOT", Some(root.to_str().unwrap()))]);
        let snapshot_dir = Paths::in_data_dir(SNAPSHOT_SUBDIR);
        let _ = std::fs::remove_dir_all(&snapshot_dir);
        let snapshot_files = || -> Vec<(PathBuf, SystemTime)> {
            let Ok(entries) = std::fs::read_dir(&snapshot_dir) else {
                return Vec::new();
            };
            entries
                .flatten()
                .filter(|entry| entry.path().extension() == Some("kv".as_ref()))
                .map(|entry| (entry.path(), entry.metadata().unwrap().modified().unwrap()))
                .collect()
        };
        let execution =
            std::env::var("GOOSE_LITERT_TEST_BACKEND").unwrap_or_else(|_| "gpu".to_string());
        let thinking = std::env::var_os("GOOSE_LITERT_TEST_THINKING").is_some();
        let settings = ModelSettings {
            context_size: Some(4096),
            enable_thinking: thinking,
            sampling: SamplingConfig::Greedy,
            litert: Some(LiteRtSettings {
                backend: Some(execution.clone()),
                speculative_decoding: None,
            }),
            ..ModelSettings::default()
        };
        let resolved = ResolvedModelPaths {
            model_path: PathBuf::from(model),
            context_limit: 4096,
            settings: settings.clone(),
            mmproj_path: None,
            backend_id: Some(LITERT_BACKEND_ID.to_string()),
            draft_model_path: None,
        };
        let backend = LiteRtBackend::new();
        let load_started = Instant::now();
        let mut loaded = backend
            .load_model("gemma-4-E2B-it", &resolved, &settings)
            .expect("model loads");
        let snapshots = loaded
            .as_any_mut()
            .downcast_mut::<LoadedModel>()
            .unwrap()
            .snapshots
            .is_some();
        eprintln!(
            "[{execution}] thinking {thinking}, snapshots {snapshots}, load {} ms",
            load_started.elapsed().as_millis()
        );

        let system = "You are a concise household assistant. Use the weather tool for any \
                      weather question, then answer in one short sentence.";
        let tools = household_tools();
        let report = |label: &str, reply: &Reply| {
            let stats = reply.stats.clone().unwrap_or_default();
            eprintln!(
                "[{execution}] {label}: prompt {:?}, ttft {:?} ms, prefill {:?} ms, reused {:?}, \
                 output {:?}, elapsed {:?} ms, text {:?}",
                reply.input_tokens,
                stats.time_to_first_token_ms,
                stats.prefill_ms,
                stats.reused_prefix_tokens,
                stats.output_tokens,
                stats.elapsed_ms,
                reply.text
            );
        };
        let reused = |reply: &Reply| reply.stats.as_ref().unwrap().reused_prefix_tokens;
        let mut ask = |system: &str, history: &[Message], tools: &[Tool], stop: bool| {
            generate_once(
                &backend,
                loaded.as_mut(),
                &resolved,
                system,
                history,
                tools,
                stop,
            )
        };

        let mut history =
            vec![Message::user().with_text("What is the weather in Paris right now?")];
        let first = ask(system, &history, &tools, false);
        report("turn 1", &first);
        assert!(first.error.is_none(), "{:?}", first.error);
        assert_eq!(reused(&first), Some(0));
        let (id, name, arguments) = first
            .tool_requests
            .first()
            .cloned()
            .expect("the model calls the weather tool");
        assert_eq!(name, "weather__get_weather");
        assert!(arguments.get("city").is_some(), "{arguments:?}");

        let mut call_message = Message::assistant();
        if !first.text.trim().is_empty() {
            call_message = call_message.with_text(first.text.clone());
        }
        history.push(call_message.with_tool_request(
            &id,
            Ok(CallToolRequestParams::new(Cow::Owned(name)).with_arguments(arguments)),
        ));
        history.push(Message::user().with_tool_response(
            &id,
            Ok(CallToolResult::success(vec![Content::text(
                r#"{"city":"Paris","condition":"sunny","temp_c":21}"#,
            )])),
        ));
        let answer = ask(system, &history, &tools, false);
        report("tool result", &answer);
        assert!(answer.error.is_none(), "{:?}", answer.error);
        assert!(answer.tool_requests.is_empty());
        assert!(!answer.text.trim().is_empty());
        assert!(reused(&answer).unwrap() > 0);

        history.push(Message::assistant().with_text(answer.text.clone()));
        let side = ask(
            "Reply with one word.",
            &[Message::user().with_text("Say yes.")],
            &[],
            false,
        );
        report("side call", &side);
        assert!(side.error.is_none(), "{:?}", side.error);
        let parked = snapshot_files();
        if snapshots {
            assert_eq!(parked.len(), 1, "the side call saves the chat: {parked:?}");
        } else {
            assert!(parked.is_empty());
        }

        // A second side call while nothing is retained, as titling follows a
        // memory pass: it stays a side call and leaves the chat's file alone.
        let titling = ask(
            "Give this conversation a title of three words.",
            &[Message::user().with_text("The weather in Paris.")],
            &[],
            false,
        );
        report("second side call", &titling);
        assert!(titling.error.is_none(), "{:?}", titling.error);
        assert_eq!(
            snapshot_files(),
            parked,
            "the second side call writes nothing"
        );

        history.push(Message::user().with_text("Is that warm or cold? One word."));
        let second = ask(system, &history, &tools, false);
        report("turn 2", &second);
        assert!(second.error.is_none(), "{:?}", second.error);
        assert!(!second.text.trim().is_empty());
        if snapshots {
            // Restored: the whole prompt was prefilled over the snapshot.
            assert_eq!(reused(&second), None);
            let restored = snapshot_files();
            assert_eq!(restored.len(), 1);
            assert!(restored[0].1 > parked[0].1, "turn 2 restores the snapshot");
        } else if !thinking {
            // With thinking on, LiteRT-LM drops the first user turn's thoughts
            // by rewinding to the checkpoint it saved before that turn: step 0.
            assert!(
                reused(&second).unwrap() > 0,
                "the side call must leave the retained conversation intact"
            );
        }

        history.push(Message::assistant().with_text(second.text.clone()));
        history.push(Message::user().with_text("And in Fahrenheit? Just the number."));
        let third = ask(system, &history, &tools, false);
        report("turn 3", &third);
        assert!(third.error.is_none(), "{:?}", third.error);
        if !thinking {
            assert!(
                reused(&third).unwrap() > 0,
                "a restored conversation extends like any other"
            );
        }

        // The same request again, as when a reply is regenerated.
        let again = ask(system, &history, &tools, false);
        report("regenerated", &again);
        assert!(again.error.is_none(), "{:?}", again.error);
        assert!(!again.text.trim().is_empty());
        assert_eq!(reused(&again), if snapshots { None } else { Some(0) });

        history.push(Message::assistant().with_text(again.text.clone()));
        history.push(Message::user().with_text("Describe Paris in three long paragraphs."));
        let cancelled = ask(system, &history, &tools, true);
        report("cancelled", &cancelled);
        assert!(cancelled.error.is_none(), "{:?}", cancelled.error);

        let recovered = ask(system, &history, &tools, false);
        report("after cancel", &recovered);
        assert!(recovered.error.is_none(), "{:?}", recovered.error);
        assert!(!recovered.text.trim().is_empty());
        // A cancelled conversation is never reused; with snapshots the chat's
        // is restored instead of prefilled from nothing.
        assert_eq!(reused(&recovered), if snapshots { None } else { Some(0) });

        history.push(Message::assistant().with_text(recovered.text.clone()));
        history.push(Message::user().with_text("Thank you. One word."));
        let last = ask(system, &history, &tools, false);
        report("after recovery", &last);
        assert!(last.error.is_none(), "{:?}", last.error);
        if !thinking {
            assert!(reused(&last).unwrap() > 0);
        }
    }
}
