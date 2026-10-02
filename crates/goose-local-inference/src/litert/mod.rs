//! LiteRT-LM backend: `.litertlm` models through the LiteRT-LM C API's
//! stateful Conversation. Each loaded model keeps one conversation as its
//! prompt cache; a request that extends it sends only the new messages.

mod convert;
mod ffi;

use std::any::Any;
use std::borrow::Cow;
use std::sync::mpsc::RecvTimeoutError;
use std::time::{Duration, Instant};

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
    engine: ffi::Engine,
    max_num_tokens: usize,
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
    /// Replace the retained conversation, history as preface.
    Recreate,
    /// Answer in a conversation of its own and leave the retained one alone.
    Throwaway,
}

/// How a request uses the retained conversation.
///
/// A different system message or tool set gets its own conversation when it
/// weighs less than half of what is retained, the rule the llama.cpp backend
/// uses for its sacrificial context: side calls such as memory extraction
/// never cost the chat its prefix, while a chat whose tools changed takes the
/// retained slot over instead of running beside it forever.
fn plan(held: Option<Held<'_>>, identity: &Identity, options: Options, messages: &[Value]) -> Plan {
    let Some(held) = held else {
        return Plan::Recreate;
    };
    if held.identity != identity {
        let request = weight(identity, messages);
        let retained = weight(held.identity, held.consumed);
        return if request.saturating_mul(2) < retained {
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
    } else {
        Plan::Recreate
    }
}

fn weight(identity: &Identity, messages: &[Value]) -> usize {
    identity.system.len()
        + identity.tools.as_ref().map_or(0, String::len)
        + messages
            .iter()
            .map(|message| message.to_string().len())
            .sum::<usize>()
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
        Ok(Box::new(LoadedModel {
            main: None,
            engine,
            max_num_tokens,
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
        let history = messages.len().checked_sub(1).ok_or_else(|| {
            ProviderError::ExecutionError("LiteRT-LM received no messages to answer".to_string())
        })?;
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
            &identity,
            options,
            &messages,
        );
        tracing::debug!(
            target: "giap::kv",
            backend = LITERT_BACKEND_ID,
            plan = ?decision,
            messages = messages.len(),
            "LiteRT-LM conversation plan"
        );

        let turn = match decision {
            Plan::Extend { from } => {
                let main = loaded
                    .main
                    .as_mut()
                    .expect("an extend plan has a retained conversation");
                let turn = run_turn(&mut main.conversation, &messages[from..], &request, false);
                if let Ok(Turn::Completed(done)) = &turn {
                    main.consumed.extend_from_slice(&messages[from..]);
                    main.consumed
                        .extend(convert::generated_message(&done.text, &done.tool_calls));
                } else {
                    loaded.main = None;
                }
                turn
            }
            Plan::Recreate => {
                loaded.main = None;
                let conversation = open_conversation(
                    &loaded.engine,
                    &identity,
                    options,
                    request.settings,
                    &messages[..history],
                )?;
                let mut main = MainConversation {
                    conversation,
                    identity,
                    options,
                    consumed: messages[..history].to_vec(),
                };
                let turn = run_turn(&mut main.conversation, &messages[history..], &request, true);
                if let Ok(Turn::Completed(done)) = &turn {
                    main.consumed.push(messages[history].clone());
                    main.consumed
                        .extend(convert::generated_message(&done.text, &done.tool_calls));
                    loaded.main = Some(main);
                }
                turn
            }
            Plan::Throwaway => {
                let mut conversation = open_conversation(
                    &loaded.engine,
                    &identity,
                    options,
                    request.settings,
                    &messages[..history],
                )?;
                run_turn(&mut conversation, &messages[history..], &request, true)
            }
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
    fn reused_prefix_tokens(&self) -> usize {
        match (self.tokens_after, &self.new_turns) {
            (Some(after), Some(turns)) => {
                let prefilled: usize = turns.prefill.iter().map(|(tokens, _)| tokens).sum();
                after
                    .saturating_sub(prefilled + turns.decode_tokens)
                    .min(self.tokens_before)
            }
            _ => self.tokens_before,
        }
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

fn run_turn(
    conversation: &mut ffi::Conversation,
    tail: &[Value],
    request: &LocalGenerationRequest<'_>,
    fresh: bool,
) -> Result<Turn, ProviderError> {
    let tokens_before = if fresh {
        0
    } else {
        conversation.token_count().unwrap_or(0)
    };
    let turns_before = conversation.turn_counts();
    let sent = Instant::now();
    let chunks = conversation.send(&convert::send_payload(tail))?;
    let mut stream = TurnStream::new(request.message_id, request.tx, sent);
    loop {
        let chunk = match chunks.recv_timeout(STREAM_POLL) {
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
                        tokens_before,
                        tokens_after: conversation.token_count().ok(),
                        new_turns: turns_before.and_then(|before| conversation.turns_since(before)),
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
        reused_prefix_tokens: Some(numbers.reused_prefix_tokens()),
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
        assert_eq!(
            plan(None, &identity("chat"), OPTIONS, &[user("hi")]),
            Plan::Recreate
        );
    }

    #[test]
    fn a_request_extending_the_conversation_sends_only_its_tail() {
        let chat = identity("chat");
        let consumed = vec![user("hi"), assistant("hello")];
        let request = vec![user("hi"), assistant("hello"), user("weather?")];
        assert_eq!(
            plan(held(&chat, &consumed), &chat, OPTIONS, &request),
            Plan::Extend { from: 2 }
        );
    }

    #[test]
    fn a_diverged_or_repeated_history_recreates() {
        let chat = identity("chat");
        let consumed = vec![user("hi"), assistant("hello")];
        let edited = vec![user("hey"), assistant("hello"), user("weather?")];
        assert_eq!(
            plan(held(&chat, &consumed), &chat, OPTIONS, &edited),
            Plan::Recreate
        );
        assert_eq!(
            plan(held(&chat, &consumed), &chat, OPTIONS, &consumed),
            Plan::Recreate
        );
        let compacted = vec![user("summary"), user("weather?")];
        assert_eq!(
            plan(held(&chat, &consumed), &chat, OPTIONS, &compacted),
            Plan::Recreate
        );
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
        assert_eq!(
            plan(held(&chat, &consumed), &chat, no_thinking, &request),
            Plan::Recreate
        );
    }

    #[test]
    fn a_small_side_call_runs_beside_the_retained_conversation() {
        let chat = Identity {
            system: "chat ".repeat(500),
            tools: Some("tool ".repeat(500)),
        };
        let consumed = vec![user("hi"), assistant("hello")];
        let extraction = Identity {
            system: "Extract memories.".to_string(),
            tools: None,
        };
        assert_eq!(
            plan(
                held(&chat, &consumed),
                &extraction,
                OPTIONS,
                &[user("the user said hi")]
            ),
            Plan::Throwaway
        );
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
        assert_eq!(
            plan(held(&chat, &consumed), &retooled, OPTIONS, &request),
            Plan::Recreate
        );
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
        };
        assert_eq!(extended.prefill_ms(), Some(200));
        assert_eq!(extended.reused_prefix_tokens(), 1000);
        assert_eq!(extended.prompt_tokens(), 1200);
        assert_eq!(extended.decode_tokens(), Some(50));

        let rewound = TurnNumbers {
            tokens_before: 1000,
            tokens_after: Some(1250),
            new_turns: Some(ffi::NewTurns {
                prefill: vec![(600, 1200.0), (200, 400.0)],
                decode_tokens: 50,
            }),
        };
        assert_eq!(rewound.reused_prefix_tokens(), 400);
        assert_eq!(rewound.prefill_ms(), Some(1000));

        let unmeasured = TurnNumbers {
            tokens_before: 300,
            tokens_after: None,
            new_turns: None,
        };
        assert_eq!(unmeasured.prefill_ms(), None);
        assert_eq!(unmeasured.reused_prefix_tokens(), 300);
        assert_eq!(unmeasured.prompt_tokens(), 300);
    }

    struct Reply {
        text: String,
        tool_requests: Vec<(String, String, serde_json::Map<String, Value>)>,
        stats: Option<ProviderStats>,
        error: Option<ProviderError>,
    }

    fn weather_tool() -> Tool {
        Tool::new(
            "weather__get_weather",
            "Get the current weather for a city.",
            std::sync::Arc::new(
                json!({
                    "type": "object",
                    "properties": {"city": {"type": "string", "description": "City name"}},
                    "required": ["city"],
                })
                .as_object()
                .cloned()
                .unwrap(),
            ),
        )
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
                            reply.stats = usage.stats;
                        }
                        if stop_after_first_piece {
                            break;
                        }
                    }
                    Ok((None, Some(ProviderUsage { stats, .. }))) => reply.stats = stats,
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

    /// A real conversation with a tool round, a side call and a cancelled
    /// turn. Run with `cargo test -p goose-local-inference -- --ignored litert`,
    /// `GOOSE_LITERT_LIB_DIR` and `GOOSE_LITERT_TEST_MODEL` set.
    /// `GOOSE_LITERT_TEST_BACKEND` picks `gpu` (default) or `cpu`;
    /// `GOOSE_LITERT_TEST_THINKING` turns thinking on.
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
        eprintln!(
            "[{execution}] thinking {thinking}, load {} ms",
            load_started.elapsed().as_millis()
        );

        let system = "You are a concise household assistant. Use the weather tool for any \
                      weather question, then answer in one short sentence.";
        let tools = vec![weather_tool()];
        let report = |label: &str, reply: &Reply| {
            let stats = reply.stats.clone().unwrap_or_default();
            eprintln!(
                "[{execution}] {label}: ttft {:?} ms, prefill {:?} ms, reused {:?}, output {:?}, \
                 elapsed {:?} ms, text {:?}",
                stats.time_to_first_token_ms,
                stats.prefill_ms,
                stats.reused_prefix_tokens,
                stats.output_tokens,
                stats.elapsed_ms,
                reply.text
            );
        };

        let mut history =
            vec![Message::user().with_text("What is the weather in Paris right now?")];
        let first = generate_once(
            &backend,
            loaded.as_mut(),
            &resolved,
            system,
            &history,
            &tools,
            false,
        );
        report("turn 1", &first);
        assert!(first.error.is_none(), "{:?}", first.error);
        assert_eq!(first.stats.as_ref().unwrap().reused_prefix_tokens, Some(0));
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
        let answer = generate_once(
            &backend,
            loaded.as_mut(),
            &resolved,
            system,
            &history,
            &tools,
            false,
        );
        report("tool result", &answer);
        assert!(answer.error.is_none(), "{:?}", answer.error);
        assert!(answer.tool_requests.is_empty());
        assert!(!answer.text.trim().is_empty());
        assert!(answer.stats.as_ref().unwrap().reused_prefix_tokens.unwrap() > 0);

        history.push(Message::assistant().with_text(answer.text.clone()));
        let side = generate_once(
            &backend,
            loaded.as_mut(),
            &resolved,
            "Reply with one word.",
            &[Message::user().with_text("Say yes.")],
            &[],
            false,
        );
        report("side call", &side);
        assert!(side.error.is_none(), "{:?}", side.error);

        history.push(Message::user().with_text("Is that warm or cold? One word."));
        let second = generate_once(
            &backend,
            loaded.as_mut(),
            &resolved,
            system,
            &history,
            &tools,
            false,
        );
        report("turn 2", &second);
        assert!(second.error.is_none(), "{:?}", second.error);
        assert!(!second.text.trim().is_empty());
        // With thinking on, LiteRT-LM drops the first user turn's thoughts by
        // rewinding to the checkpoint it saved before that turn, which is step 0.
        if !thinking {
            assert!(
                second.stats.as_ref().unwrap().reused_prefix_tokens.unwrap() > 0,
                "the side call must leave the retained conversation intact"
            );
        }

        history.push(Message::assistant().with_text(second.text.clone()));
        history.push(Message::user().with_text("And in Fahrenheit? Just the number."));
        let third = generate_once(
            &backend,
            loaded.as_mut(),
            &resolved,
            system,
            &history,
            &tools,
            false,
        );
        report("turn 3", &third);
        assert!(third.error.is_none(), "{:?}", third.error);
        assert!(third.stats.as_ref().unwrap().reused_prefix_tokens.unwrap() > 0);

        history.push(Message::assistant().with_text(third.text.clone()));
        history.push(Message::user().with_text("Describe Paris in three long paragraphs."));
        let cancelled = generate_once(
            &backend,
            loaded.as_mut(),
            &resolved,
            system,
            &history,
            &tools,
            true,
        );
        report("cancelled", &cancelled);
        assert!(cancelled.error.is_none(), "{:?}", cancelled.error);

        let recovered = generate_once(
            &backend,
            loaded.as_mut(),
            &resolved,
            system,
            &history,
            &tools,
            false,
        );
        report("after cancel", &recovered);
        assert!(recovered.error.is_none(), "{:?}", recovered.error);
        assert!(!recovered.text.trim().is_empty());
        assert_eq!(
            recovered.stats.as_ref().unwrap().reused_prefix_tokens,
            Some(0),
            "a cancelled conversation is recreated, not reused"
        );
    }
}
