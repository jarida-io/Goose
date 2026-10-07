//! Pure translation between goose's conversation types and the JSON the
//! LiteRT-LM Conversation API reads and streams.

use std::collections::HashMap;

use goose_provider_types::conversation::message::{Message, MessageContent, ToolResponse};
use goose_provider_types::errors::ProviderError;
use goose_provider_types::formats::openai::format_tools;
use rmcp::model::{CallToolRequestParams, ContentBlock, ErrorData, ResourceContents, Role, Tool};
use serde_json::{json, Map, Value};

const IMAGE_NOT_SUPPORTED: &str =
    "[Image attached — image input is not supported with the currently selected model]";
const UNPARSEABLE_TOOL_CALL: &str = "unparseable_tool_call";
const TOOL_CALL_ID_PREFIX: &str = "lrt_";
const ROUND_HEX_DIGITS: usize = 16;

#[derive(Debug, Clone, PartialEq)]
pub(super) struct ToolCall {
    pub(super) name: String,
    pub(super) arguments: Map<String, Value>,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) enum ChunkEvent {
    TextDelta(String),
    ThoughtDelta(String),
    ToolCalls(Vec<ToolCall>),
    Final,
    Error(String),
}

/// Tool declarations in the OpenAI-wrapped form Gemma 4's template reads
/// (`tool['function']['name']`); `None` when there are no tools.
pub(super) fn tools_json(tools: &[Tool]) -> Result<Option<String>, ProviderError> {
    if tools.is_empty() {
        return Ok(None);
    }
    let declarations = format_tools(tools).map_err(|error| {
        ProviderError::ExecutionError(format!("LiteRT-LM tool declarations: {error}"))
    })?;
    Ok(Some(Value::Array(declarations).to_string()))
}

/// The system message as the JSON content `conversation_config_set_system_message` expects.
pub(super) fn system_message_json(system: &str) -> Option<String> {
    (!system.trim().is_empty()).then(|| Value::String(system.to_string()).to_string())
}

/// Id for the `index`-th tool call of one generation. Every call of a
/// generation shares `round`, which is how the split messages goose stores per
/// call are folded back into the single message the model produced.
pub(super) fn tool_call_id(round: u64, index: usize) -> String {
    format!("{TOOL_CALL_ID_PREFIX}{round:016x}_{index}")
}

fn round_of(id: &str) -> Option<&str> {
    let (round, index) = id.strip_prefix(TOOL_CALL_ID_PREFIX)?.split_once('_')?;
    let well_formed = round.len() == ROUND_HEX_DIGITS
        && round.bytes().all(|b| b.is_ascii_hexdigit())
        && !index.is_empty()
        && index.bytes().all(|b| b.is_ascii_digit());
    well_formed.then_some(round)
}

/// goose messages as LiteRT-LM messages, in one canonical form used both for
/// sending and for comparing a request against what a conversation already
/// holds. Thinking is never replayed; ids do not appear.
pub(super) fn litert_messages(messages: &[Message]) -> Vec<Value> {
    let tool_names = tool_names_by_id(messages);
    let mut builder = Builder::default();
    for message in messages.iter().filter(|message| message.is_agent_visible()) {
        match message.role {
            Role::Assistant => builder.assistant(message),
            Role::User => builder.user(message, &tool_names),
        }
    }
    builder.messages
}

/// The canonical form of an assistant turn the model generated, matching what
/// [`litert_messages`] produces once goose sends that turn back.
pub(super) fn generated_message(text: &str, tool_calls: &[ToolCall]) -> Option<Value> {
    let calls = tool_calls
        .iter()
        .map(|call| function_json(&call.name, Value::Object(call.arguments.clone())))
        .collect();
    assistant_json(text.trim(), calls)
}

/// The payload for `conversation_send_message_stream`: a single message as an
/// object, several as an array. A lone user message must be an object, because
/// that is what LiteRT-LM's channel-content filtering keys on.
pub(super) fn send_payload(messages: &[Value]) -> String {
    match messages {
        [message] => message.to_string(),
        _ => Value::Array(messages.to_vec()).to_string(),
    }
}

pub(super) fn chunk_events(
    text: Option<&str>,
    is_final: bool,
    error: Option<&str>,
) -> Vec<ChunkEvent> {
    if let Some(error) = error {
        return vec![ChunkEvent::Error(error.to_string())];
    }
    let mut events = text
        .filter(|text| !text.is_empty())
        .map(message_events)
        .unwrap_or_default();
    if is_final {
        events.push(ChunkEvent::Final);
    }
    events
}

pub(super) fn stream_error(message: &str) -> ProviderError {
    if is_context_overflow(message) {
        ProviderError::ContextLengthExceeded(format!("LiteRT-LM: {message}"))
    } else {
        ProviderError::ExecutionError(format!("LiteRT-LM generation failed: {message}"))
    }
}

pub(super) fn is_context_overflow(message: &str) -> bool {
    message.contains("Input token ids are too long")
        || message.contains("context window out of bounds")
}

fn tool_names_by_id(messages: &[Message]) -> HashMap<&str, &str> {
    let mut names = HashMap::new();
    for content in messages.iter().flat_map(|message| &message.content) {
        let (id, call) = match content {
            MessageContent::ToolRequest(request) => (request.id.as_str(), &request.tool_call),
            MessageContent::FrontendToolRequest(request) => {
                (request.id.as_str(), &request.tool_call)
            }
            _ => continue,
        };
        let name = match call {
            Ok(call) => call.name.as_ref(),
            Err(_) => UNPARSEABLE_TOOL_CALL,
        };
        names.insert(id, name);
    }
    names
}

#[derive(Default)]
struct Builder {
    messages: Vec<Value>,
    /// The generation round of each assistant message's tool calls.
    rounds: Vec<Option<String>>,
}

impl Builder {
    fn assistant(&mut self, message: &Message) {
        let mut texts = Vec::new();
        let mut calls = Vec::new();
        let mut ids = Vec::new();
        for content in &message.content {
            match content {
                MessageContent::Text(text) if !text.text.trim().is_empty() => {
                    texts.push(text.text.as_str());
                }
                MessageContent::ToolRequest(request) => {
                    calls.push(tool_call_json(&request.tool_call));
                    ids.push(request.id.as_str());
                }
                MessageContent::FrontendToolRequest(request) => {
                    calls.push(tool_call_json(&request.tool_call));
                    ids.push(request.id.as_str());
                }
                _ => {}
            }
        }
        let text = texts.join("\n");
        let text = text.trim();
        let round = shared_round(&ids);

        if text.is_empty() && !calls.is_empty() {
            if let Some(index) = self.open_round(round.as_deref()) {
                extend_tool_calls(&mut self.messages[index], calls);
                return;
            }
        }
        if let Some(last) = self
            .messages
            .last_mut()
            .filter(|last| last["role"] == "assistant")
        {
            let had_calls = last.get("tool_calls").is_some();
            merge_assistant(last, text, calls);
            if let Some(previous) = self.rounds.last_mut() {
                *previous = if !had_calls || *previous == round {
                    round
                } else {
                    None
                };
            }
            return;
        }
        if let Some(message) = assistant_json(text, calls) {
            self.push(message, round);
        }
    }

    fn user(&mut self, message: &Message, tool_names: &HashMap<&str, &str>) {
        let mut texts = Vec::new();
        for content in &message.content {
            match content {
                MessageContent::Text(text) if !text.text.trim().is_empty() => {
                    texts.push(text.text.clone());
                }
                MessageContent::Image(_) => texts.push(IMAGE_NOT_SUPPORTED.to_string()),
                MessageContent::ToolResponse(response) => {
                    let name = tool_names.get(response.id.as_str()).copied();
                    self.push(tool_message(name, response), None);
                }
                _ => {}
            }
        }
        let text = texts.join("\n");
        let text = text.trim();
        if !text.is_empty() {
            self.push(
                json!({"role": "user", "content": [{"type": "text", "text": text}]}),
                None,
            );
        }
    }

    fn push(&mut self, message: Value, round: Option<String>) {
        self.messages.push(message);
        self.rounds.push(round);
    }

    /// The assistant message that opened `round`, if everything after it is
    /// that round's tool responses.
    fn open_round(&self, round: Option<&str>) -> Option<usize> {
        let round = round?;
        let index = self
            .messages
            .iter()
            .rposition(|message| message["role"] != "tool")?;
        (self.messages[index]["role"] == "assistant"
            && self.rounds[index].as_deref() == Some(round))
        .then_some(index)
    }
}

fn shared_round(ids: &[&str]) -> Option<String> {
    let (first, rest) = ids.split_first()?;
    let round = round_of(first)?;
    rest.iter()
        .all(|id| round_of(id) == Some(round))
        .then(|| round.to_string())
}

fn assistant_json(text: &str, calls: Vec<Value>) -> Option<Value> {
    if text.is_empty() && calls.is_empty() {
        return None;
    }
    let mut message = json!({"role": "assistant"});
    if !text.is_empty() {
        message["content"] = json!([{"type": "text", "text": text}]);
    }
    if !calls.is_empty() {
        message["tool_calls"] = Value::Array(calls);
    }
    Some(message)
}

fn merge_assistant(message: &mut Value, text: &str, calls: Vec<Value>) {
    if !text.is_empty() {
        let merged = match message["content"][0]["text"].as_str() {
            Some(existing) => format!("{existing}\n{text}"),
            None => text.to_string(),
        };
        message["content"] = json!([{"type": "text", "text": merged}]);
    }
    extend_tool_calls(message, calls);
}

fn extend_tool_calls(message: &mut Value, calls: Vec<Value>) {
    if calls.is_empty() {
        return;
    }
    match message.get_mut("tool_calls").and_then(Value::as_array_mut) {
        Some(existing) => existing.extend(calls),
        None => message["tool_calls"] = Value::Array(calls),
    }
}

fn tool_call_json(call: &Result<CallToolRequestParams, ErrorData>) -> Value {
    match call {
        Ok(call) => function_json(
            &call.name,
            Value::Object(call.arguments.clone().unwrap_or_default()),
        ),
        Err(_) => function_json(UNPARSEABLE_TOOL_CALL, json!({})),
    }
}

fn function_json(name: &str, arguments: Value) -> Value {
    json!({"type": "function", "function": {"name": name, "arguments": arguments}})
}

fn tool_message(name: Option<&str>, response: &ToolResponse) -> Value {
    let mut item = json!({"type": "tool_response", "response": tool_response_payload(response)});
    if let Some(name) = name {
        item["name"] = json!(name);
    }
    json!({"role": "tool", "content": [item]})
}

/// A tool result as the template's `response`: a JSON object when the tool
/// returned one, its text otherwise.
fn tool_response_payload(response: &ToolResponse) -> Value {
    let result = match &response.tool_result {
        Ok(result) => result,
        Err(error) => {
            return Value::String(format!(
                "The tool call returned the following error:\n{error}"
            ))
        }
    };
    let text = result
        .content
        .iter()
        .filter_map(|content| match content {
            ContentBlock::Text(text) => Some(text.text.clone()),
            ContentBlock::Image(_) => Some(IMAGE_NOT_SUPPORTED.to_string()),
            ContentBlock::Resource(resource) => match &resource.resource {
                ResourceContents::TextResourceContents { text, .. } => Some(text.clone()),
                ResourceContents::BlobResourceContents { .. } => None,
                _ => {
                    tracing::warn!("Unsupported resource content in LiteRT tool response");
                    None
                }
            },
            _ => None,
        })
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    match serde_json::from_str::<Value>(text.trim()) {
        Ok(object @ Value::Object(_)) => object,
        _ => Value::String(text),
    }
}

fn message_events(chunk: &str) -> Vec<ChunkEvent> {
    let Ok(Value::Object(message)) = serde_json::from_str::<Value>(chunk) else {
        return vec![ChunkEvent::TextDelta(chunk.to_string())];
    };
    let mut events = Vec::new();
    let thought = message
        .get("reasoning_content")
        .and_then(Value::as_str)
        .or_else(|| {
            message
                .get("channels")
                .and_then(|channels| channels.get("thought"))
                .and_then(Value::as_str)
        });
    if let Some(thought) = thought.filter(|thought| !thought.is_empty()) {
        events.push(ChunkEvent::ThoughtDelta(thought.to_string()));
    }
    let text = content_text(message.get("content"));
    if !text.is_empty() {
        events.push(ChunkEvent::TextDelta(text));
    }
    let mut calls: Vec<ToolCall> = message
        .get("tool_calls")
        .and_then(Value::as_array)
        .map(|calls| calls.iter().filter_map(parse_tool_call).collect())
        .unwrap_or_default();
    calls.extend(rejected_calls(message.get("content")));
    if !calls.is_empty() {
        events.push(ChunkEvent::ToolCalls(calls));
    }
    events
}

fn content_text(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Array(parts)) => parts.iter().map(part_text).collect(),
        Some(part @ Value::Object(_)) => part_text(part).to_string(),
        _ => String::new(),
    }
}

/// A part's text; a tool call the engine's parser rejected is not text, see [`rejected_calls`].
fn part_text(part: &Value) -> &str {
    let is_text = part
        .get("type")
        .and_then(Value::as_str)
        .is_none_or(|kind| kind == "text");
    if is_text && part.get("error").is_none() {
        part.get("text").and_then(Value::as_str).unwrap_or_default()
    } else {
        ""
    }
}

/// Tool calls the engine's parser rejected, which a conversation that keeps them returns as text
/// parts with an "error" field, read leniently. One still unreadable is left out and logged.
fn rejected_calls(content: Option<&Value>) -> Vec<ToolCall> {
    let Some(Value::Array(parts)) = content else {
        return Vec::new();
    };
    parts
        .iter()
        .filter(|part| part.get("error").is_some())
        .filter_map(|part| {
            let block = part.get("text").and_then(Value::as_str).unwrap_or_default();
            match super::fc_repair::parse_call(block) {
                Some((name, arguments)) => {
                    tracing::info!(
                        backend = "litert",
                        tool = %name,
                        "Read a tool call LiteRT-LM's parser rejected"
                    );
                    Some(ToolCall { name, arguments })
                }
                None => {
                    tracing::warn!(
                        backend = "litert",
                        block,
                        "Left out a tool call LiteRT-LM's parser rejected and that could not be read"
                    );
                    None
                }
            }
        })
        .collect()
}

fn parse_tool_call(call: &Value) -> Option<ToolCall> {
    let function = call.get("function").unwrap_or(call);
    let name = function.get("name").and_then(Value::as_str)?.trim();
    if name.is_empty() {
        return None;
    }
    let arguments = match function.get("arguments") {
        Some(Value::Object(arguments)) => arguments.clone(),
        Some(Value::String(arguments)) => match serde_json::from_str(arguments) {
            Ok(Value::Object(arguments)) => arguments,
            _ => Map::new(),
        },
        _ => Map::new(),
    };
    Some(ToolCall {
        name: name.to_string(),
        arguments,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rmcp::model::{CallToolResult, ContentBlock};
    use std::borrow::Cow;

    fn args(value: Value) -> Map<String, Value> {
        value.as_object().cloned().unwrap()
    }

    fn call(name: &str, arguments: Value) -> CallToolRequestParams {
        CallToolRequestParams::new(Cow::Owned(name.to_string())).with_arguments(args(arguments))
    }

    fn weather_tool() -> Tool {
        Tool::new(
            "weather__get_weather",
            "Current weather for a city.",
            std::sync::Arc::new(args(json!({
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            }))),
        )
    }

    fn text_result(text: &str) -> Result<CallToolResult, ErrorData> {
        Ok(CallToolResult::success(vec![ContentBlock::text(text)]))
    }

    #[test]
    fn a_tool_call_the_engine_rejected_is_read_and_kept_out_of_the_text() {
        let chunk = json!({
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Saving that. "},
                {
                    "type": "text",
                    "text": "call:giap-memory__save_memory{content:<|\"|>Wanjiru is seven.<|\"|>,segment:knowledge}",
                    "error": "Failed to parse tool calls: NoViableAltError"
                },
                {"type": "text", "text": "call:!!", "error": "Failed to parse tool calls: ..."}
            ]
        })
        .to_string();
        let events = chunk_events(Some(&chunk), false, None);
        assert_eq!(
            events,
            vec![
                ChunkEvent::TextDelta("Saving that. ".to_string()),
                ChunkEvent::ToolCalls(vec![ToolCall {
                    name: "giap-memory__save_memory".to_string(),
                    arguments: json!({"content": "Wanjiru is seven.", "segment": "knowledge"})
                        .as_object()
                        .cloned()
                        .unwrap(),
                }]),
            ]
        );
    }
    #[test]
    fn tools_use_the_openai_wrapped_form() {
        let tools = tools_json(&[weather_tool()]).unwrap().unwrap();
        let tools: Value = serde_json::from_str(&tools).unwrap();
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[0]["function"]["name"], "weather__get_weather");
        assert_eq!(
            tools[0]["function"]["description"],
            "Current weather for a city."
        );
        assert_eq!(tools[0]["function"]["parameters"]["required"][0], "city");
        assert_eq!(tools_json(&[]).unwrap(), None);
    }

    #[test]
    fn system_message_is_a_json_string() {
        assert_eq!(
            system_message_json("You are \"Pond\".").as_deref(),
            Some(r#""You are \"Pond\".""#)
        );
        assert_eq!(system_message_json("  \n"), None);
    }

    #[test]
    fn a_tool_round_converts_to_template_messages() {
        let messages = vec![
            Message::user().with_text("Weather in Paris?"),
            Message::assistant()
                .with_thinking("The user wants weather.", "")
                .with_text("Checking. ")
                .with_tool_request(
                    "call-1",
                    Ok(call("weather__get_weather", json!({"city": "Paris"}))),
                ),
            Message::user().with_tool_response("call-1", text_result(r#"{"temp_c": 21}"#)),
        ];

        assert_eq!(
            litert_messages(&messages),
            vec![
                json!({"role": "user", "content": [{"type": "text", "text": "Weather in Paris?"}]}),
                json!({
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Checking."}],
                    "tool_calls": [{"type": "function", "function": {
                        "name": "weather__get_weather", "arguments": {"city": "Paris"}}}],
                }),
                json!({"role": "tool", "content": [{
                    "type": "tool_response", "name": "weather__get_weather",
                    "response": {"temp_c": 21}}]}),
            ]
        );
    }

    #[test]
    fn tool_results_that_are_not_json_objects_stay_text() {
        let messages = vec![
            Message::assistant()
                .with_tool_request("a", Ok(call("memory__recall", json!({}))))
                .with_tool_request("b", Err(ErrorData::invalid_params("bad", None))),
            Message::user()
                .with_tool_response("a", text_result("[1, 2]"))
                .with_tool_response("b", Err(ErrorData::internal_error("boom", None)))
                .with_tool_response("orphan", text_result("plain")),
        ];
        let converted = litert_messages(&messages);

        assert_eq!(
            converted[0]["tool_calls"][1]["function"],
            json!({"name": "unparseable_tool_call", "arguments": {}})
        );
        assert_eq!(converted[1]["content"][0]["name"], "memory__recall");
        assert_eq!(converted[1]["content"][0]["response"], "[1, 2]");
        assert_eq!(converted[2]["content"][0]["name"], "unparseable_tool_call");
        assert!(converted[2]["content"][0]["response"]
            .as_str()
            .unwrap()
            .starts_with("The tool call returned the following error:\n"));
        assert!(converted[3]["content"][0].get("name").is_none());
        assert_eq!(converted[3]["content"][0]["response"], "plain");
    }

    #[test]
    fn images_become_a_note_and_hidden_messages_are_skipped() {
        let messages = vec![
            Message::user()
                .with_text("What is this?")
                .with_image("aGVsbG8=", "image/png"),
            Message::user()
                .with_text("internal nudge")
                .with_visibility(true, false),
        ];
        assert_eq!(
            litert_messages(&messages),
            vec![json!({"role": "user", "content": [{"type": "text",
                "text": format!("What is this?\n{IMAGE_NOT_SUPPORTED}")}]})]
        );
    }

    /// One generation with two calls, stored by goose's agent as interleaved
    /// request/response pairs, folds back into the single message the model
    /// produced. Calls from separate generations stay apart.
    #[test]
    fn split_tool_rounds_fold_back_into_one_message() {
        let round = 0x00ab_cdef_0123_4567;
        let first = tool_call_id(round, 0);
        let second = tool_call_id(round, 1);
        let paris = call("weather__get_weather", json!({"city": "Paris"}));
        let rome = call("weather__get_weather", json!({"city": "Rome"}));
        let stored = vec![
            Message::user().with_text("Paris and Rome?"),
            Message::assistant()
                .with_thinking("two cities", "")
                .with_tool_request(&first, Ok(paris.clone())),
            Message::user().with_tool_response(&first, text_result("sunny")),
            Message::assistant()
                .with_thinking("two cities", "")
                .with_tool_request(&second, Ok(rome.clone())),
            Message::user().with_tool_response(&second, text_result("rain")),
        ];
        let converted = litert_messages(&stored);
        assert_eq!(converted.len(), 4);
        assert_eq!(converted[1]["tool_calls"].as_array().unwrap().len(), 2);
        assert_eq!(converted[2]["content"][0]["response"], "sunny");
        assert_eq!(converted[3]["content"][0]["response"], "rain");

        let separate = vec![
            stored[0].clone(),
            Message::assistant().with_tool_request(tool_call_id(1, 0), Ok(paris)),
            Message::user().with_tool_response(tool_call_id(1, 0), text_result("sunny")),
            Message::assistant().with_tool_request(tool_call_id(2, 0), Ok(rome)),
            Message::user().with_tool_response(tool_call_id(2, 0), text_result("rain")),
        ];
        assert_eq!(litert_messages(&separate).len(), 5);
    }

    /// What the backend records for a generation has to equal what the
    /// conversion makes of goose's stored copy, or every turn looks diverged.
    #[test]
    fn a_generated_turn_matches_goose_stored_copy() {
        let round = 7;
        let calls = vec![
            ToolCall {
                name: "weather__get_weather".to_string(),
                arguments: args(json!({"city": "Paris"})),
            },
            ToolCall {
                name: "weather__get_weather".to_string(),
                arguments: args(json!({"city": "Rome"})),
            },
        ];
        let recorded = generated_message("\nLet me check.  ", &calls).unwrap();

        let mut stored = vec![Message::assistant()
            .with_text("\nLet me check.")
            .with_tool_request(
                tool_call_id(round, 0),
                Ok(
                    CallToolRequestParams::new(Cow::Owned(calls[0].name.clone()))
                        .with_arguments(calls[0].arguments.clone()),
                ),
            )];
        stored.push(Message::user().with_tool_response(tool_call_id(round, 0), text_result("a")));
        stored.push(
            Message::assistant().with_tool_request(
                tool_call_id(round, 1),
                Ok(
                    CallToolRequestParams::new(Cow::Owned(calls[1].name.clone()))
                        .with_arguments(calls[1].arguments.clone()),
                ),
            ),
        );
        stored.push(Message::user().with_tool_response(tool_call_id(round, 1), text_result("b")));

        assert_eq!(litert_messages(&stored)[0], recorded);
        assert_eq!(generated_message("  ", &[]), None);
    }

    #[test]
    fn tool_call_ids_carry_their_round() {
        let id = tool_call_id(u64::MAX, 12);
        assert_eq!(id, "lrt_ffffffffffffffff_12");
        assert!(
            id.len() <= 40,
            "OpenAI rejects tool call ids over 40 characters"
        );
        assert_eq!(round_of(&id), Some("ffffffffffffffff"));
        assert_eq!(round_of("lrt_short_1"), None);
        assert_eq!(round_of("call_0123456789abcdef_1"), None);
        assert_eq!(round_of("0f1e2d3c-uuid"), None);
    }

    #[test]
    fn a_single_message_is_sent_as_an_object() {
        let user = json!({"role": "user", "content": [{"type": "text", "text": "hi"}]});
        assert_eq!(send_payload(std::slice::from_ref(&user)), user.to_string());
        let pair = vec![user.clone(), user];
        assert!(send_payload(&pair).starts_with('['));
    }

    #[test]
    fn stream_chunks_become_events() {
        assert_eq!(
            chunk_events(
                Some(r#"{"role":"assistant","content":[{"type":"text","text":"Hel"}]}"#),
                false,
                None
            ),
            vec![ChunkEvent::TextDelta("Hel".to_string())]
        );
        assert_eq!(
            chunk_events(
                Some(
                    r#"{"role":"assistant","channels":{"thought":"hm"},"reasoning_content":"hm"}"#
                ),
                false,
                None
            ),
            vec![ChunkEvent::ThoughtDelta("hm".to_string())]
        );
        assert_eq!(
            chunk_events(
                Some(r#"{"role":"assistant","channels":{"thought":"only channel"}}"#),
                false,
                None
            ),
            vec![ChunkEvent::ThoughtDelta("only channel".to_string())]
        );
        assert_eq!(
            chunk_events(
                Some(
                    r#"{"role":"assistant","tool_calls":[{"type":"function","function":{"name":"weather__get_weather","arguments":{"city":"Paris"}}}]}"#
                ),
                false,
                None
            ),
            vec![ChunkEvent::ToolCalls(vec![ToolCall {
                name: "weather__get_weather".to_string(),
                arguments: args(json!({"city": "Paris"})),
            }])]
        );
        assert_eq!(chunk_events(None, true, None), vec![ChunkEvent::Final]);
        assert_eq!(
            chunk_events(None, true, Some("CANCELLED: Task cancelled")),
            vec![ChunkEvent::Error("CANCELLED: Task cancelled".to_string())]
        );
        assert_eq!(
            chunk_events(Some("not json"), false, None),
            vec![ChunkEvent::TextDelta("not json".to_string())]
        );
        assert!(chunk_events(
            Some(r#"{"role":"assistant","content":[{"type":"text","text":""}]}"#),
            false,
            None
        )
        .is_empty());
    }

    #[test]
    fn context_overflow_is_reported_as_such() {
        let overflow = "INVALID_ARGUMENT: Input token ids are too long. Exceeding the maximum \
                        number of tokens allowed: 5000 >= 4096";
        assert!(matches!(
            stream_error(overflow),
            ProviderError::ContextLengthExceeded(_)
        ));
        assert!(matches!(
            stream_error("INTERNAL: Task failed with state: 6"),
            ProviderError::ExecutionError(_)
        ));
    }
}
