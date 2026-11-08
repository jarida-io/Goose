use rmcp::model::Tool;
use std::any::Any;

use crate::local_model_registry::ModelSettings;
use goose_provider_types::conversation::message::Message;
use goose_provider_types::errors::ProviderError;
use goose_provider_types::request_log::RequestLogHandle;

use super::{ResolvedModelPaths, StreamSender};

pub(super) trait BackendLoadedModel: Send {
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

#[cfg_attr(not(feature = "mlx"), allow(dead_code))]
pub(super) struct LocalGenerationRequest<'a> {
    pub model_name: String,
    pub system: &'a str,
    pub messages: &'a [Message],
    pub tools: &'a [Tool],
    pub settings: &'a ModelSettings,
    pub temperature: Option<f32>,
    pub max_tokens: Option<i32>,
    pub context_limit: usize,
    pub model_load_ms: Option<u64>,
    pub resolved_model: &'a ResolvedModelPaths,
    pub draft_model_path: Option<std::path::PathBuf>,
    pub message_id: &'a str,
    pub tx: &'a StreamSender,
    pub log: &'a mut Option<Box<dyn RequestLogHandle>>,
    /// The answer replaces the history of the conversation it was asked about, as a compaction's
    /// summary does (see `request_context::replacing_history`).
    pub replaces_history: bool,
    /// The conversation (goose session) the call is made for, when the caller said
    /// (see `request_context::for_session`).
    pub session_id: Option<String>,
    /// Only a tool call from this generation is used (see `request_context::tool_calls_only`). The
    /// llama.cpp native-tools path stops at the first prose. LiteRT-LM ignores it, because it can
    /// end a turn only by cancelling it, and a cancelled conversation is not kept. The
    /// emulated-tools path ignores it too, because a tool call starts there as text.
    pub tool_calls_only: bool,
    /// Made for a helper (see `request_context::as_helper`): on llama.cpp it takes the model's one
    /// context instead of a second beside the prime's. LiteRT-LM ignores it.
    pub helper: bool,
    /// A live turn waits on this helper, so llama.cpp sets that turn's cache aside on disk until
    /// the turn continues, instead of rebuilding it afterwards.
    pub prime_waiting: bool,
    /// The shape a prefill compile names (see `request_context::compiling_prefill`); `None` for
    /// any other call. LiteRT-LM ignores it.
    pub compile_prefill: Option<String>,
    /// Helper snapshots unused for longer are discarded by this compile
    /// (see `request_context::compiling_prefill_retaining`).
    pub prefill_retention_secs: Option<u64>,
    /// The compile rebuilds even a valid snapshot (see `request_context::rebuilding_prefill`).
    pub rebuild_prefill: bool,
    /// Ends the generation at its next token once true (see `request_context::stopping_when`).
    /// LiteRT-LM ignores it.
    pub stop_when: Option<goose_provider_types::request_context::StopWhen>,
}

pub(super) trait LocalInferenceBackend: Send + Sync {
    fn id(&self) -> &'static str;

    fn load_model(
        &self,
        model_id: &str,
        resolved: &ResolvedModelPaths,
        settings: &ModelSettings,
    ) -> Result<Box<dyn BackendLoadedModel>, ProviderError>;

    fn generate(
        &self,
        loaded: &mut dyn BackendLoadedModel,
        request: LocalGenerationRequest<'_>,
    ) -> Result<(), ProviderError>;

    fn available_memory_bytes(&self) -> u64;
}
