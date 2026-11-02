//! A tool call the model wrote as text instead of in its native wrapper. llama.cpp's parser
//! streams it as content, so the call never runs; Gemma 4 does this with thinking off, as
//! `<NAME key="value" />`, or as its own `NAME{key:value}` grammar (or `NAME(key="value")`)
//! behind a tag it makes up, such as `<call>`, `<tool_call>` or `<execute_tool>`, or inside
//! `[...]`, or as `<NAME>text</NAME>` with the text as the tool's one text parameter. This reads
//! such a call out of the visible content when it names a declared tool, and holds back only
//! text that could still become one.

use serde_json::{Map, Value};

use crate::litert::fc_repair::{arguments_end, parse_arguments};

/// Held text past this is released: no written call has come near it.
const MAX_HELD_BYTES: usize = 4096;

#[derive(Debug, PartialEq)]
pub(super) struct WrittenCall {
    pub name: String,
    pub arguments: Map<String, Value>,
}

pub(super) struct WrittenToolCalls {
    tools: Vec<Declared>,
    held: String,
    call: Option<WrittenCall>,
}

struct Declared {
    name: String,
    /// The parameter `<NAME>text</NAME>` fills ([`text_parameter`]).
    text_parameter: Option<String>,
}

enum Read {
    Call(WrittenCall),
    Unfinished,
    NotACall,
}

impl WrittenToolCalls {
    /// Each declared tool's name and input schema.
    pub(super) fn new<'a>(
        declared: impl IntoIterator<Item = (&'a str, &'a Map<String, Value>)>,
    ) -> Self {
        Self {
            tools: declared
                .into_iter()
                .map(|(name, schema)| Declared {
                    name: name.to_string(),
                    text_parameter: text_parameter(schema),
                })
                .collect(),
            held: String::new(),
            call: None,
        }
    }

    /// Takes the next visible text and returns what can be shown now. Once a call is read,
    /// nothing after it is: the rest of the generation would be the model guessing its result.
    pub(super) fn push(&mut self, text: &str) -> String {
        self.scan(text, false)
    }

    /// The generation ended: a call still open is read as far as it got.
    pub(super) fn finish(&mut self) -> String {
        self.scan("", true)
    }

    pub(super) fn found(&self) -> bool {
        self.call.is_some()
    }

    pub(super) fn take(&mut self) -> Option<WrittenCall> {
        self.call.take()
    }

    fn scan(&mut self, text: &str, at_end: bool) -> String {
        if self.call.is_some() {
            return String::new();
        }
        if self.tools.is_empty() {
            return text.to_string();
        }
        self.held.push_str(text);
        let mut from = 0;
        while let Some(offset) = self.held.get(from..).and_then(|rest| rest.find(['<', '['])) {
            let at = from + offset;
            let after = self.held.get(at + 1..).unwrap_or_default();
            let read = if self.held.as_bytes().get(at) == Some(&b'[') {
                self.read_fenced(after, None, at_end)
            } else {
                self.read(after, at_end)
            };
            match read {
                Read::Call(call) => {
                    self.call = Some(call);
                    self.held.truncate(at);
                    return std::mem::take(&mut self.held);
                }
                Read::Unfinished if self.held.len() - at <= MAX_HELD_BYTES => {
                    return self.held.drain(..at).collect();
                }
                _ => from = at + 1,
            }
        }
        std::mem::take(&mut self.held)
    }

    /// Reads what follows a `<`.
    fn read(&self, text: &str, at_end: bool) -> Read {
        let (word, rest) = text.split_at(word_len(text));
        if rest.is_empty() {
            return if at_end {
                Read::NotACall
            } else {
                Read::Unfinished
            };
        }
        if word.is_empty() {
            return Read::NotACall;
        }
        let Some(body) = rest.strip_prefix('>') else {
            return self.read_tag(word, rest, at_end);
        };
        let closer = format!("</{word}>");
        match self.read_fenced(body, Some(&closer), at_end) {
            Read::NotACall => self.read_body(word, body, &closer, at_end),
            read => read,
        }
    }

    /// `<NAME>text</NAME>`, NAME also as the part after `__` alone: the text fills the tool's
    /// one text parameter. Kept to one line, so a made-up tag around prose stays prose.
    fn read_body(&self, word: &str, body: &str, closer: &str, at_end: bool) -> Read {
        let Some(name) = self.resolve(word).or_else(|| self.resolve_short(word)) else {
            return Read::NotACall;
        };
        let Some(parameter) = self
            .tools
            .iter()
            .find(|tool| tool.name == name)
            .and_then(|tool| tool.text_parameter.clone())
        else {
            return Read::NotACall;
        };
        let text = match body.find(closer) {
            Some(end) => body.split_at(end).0,
            None => {
                // The part of the closing tag written so far is not the text.
                let started = (1..closer.len())
                    .rev()
                    .find(|&len| closer.get(..len).is_some_and(|part| body.ends_with(part)))
                    .unwrap_or(0);
                let text = body.split_at(body.len() - started).0;
                if !at_end {
                    return if text.trim().contains('\n') {
                        Read::NotACall
                    } else {
                        Read::Unfinished
                    };
                }
                // Cut short: only a closing tag the model began says the text was a call.
                if started < "</".len() {
                    return Read::NotACall;
                }
                text
            }
        };
        let text = text.trim();
        if text.is_empty() || text.contains('\n') {
            return Read::NotACall;
        }
        let mut arguments = Map::new();
        arguments.insert(parameter, Value::String(text.to_string()));
        Read::Call(WrittenCall { name, arguments })
    }

    /// `NAME{key:value,...}` or `NAME(key="value",...)`, optionally after `call:`, behind
    /// `<FENCE>` or `[`. Behind a fence a bare `NAME` up to `</FENCE>` is a call without
    /// arguments; behind `[` it needs arguments, so `[NAME]` in prose stays prose.
    fn read_fenced(&self, body: &str, closer: Option<&str>, at_end: bool) -> Read {
        let body = body.trim_start();
        let body = match body.strip_prefix("call:") {
            Some(rest) => rest,
            None if "call:".starts_with(body) && !at_end => return Read::Unfinished,
            None => body,
        };
        let (written, rest) = body.split_at(word_len(body));
        if rest.is_empty() && !at_end {
            return Read::Unfinished;
        }
        let Some(name) = self.resolve(written) else {
            return Read::NotACall;
        };
        let rest = rest.trim_start();
        if rest.starts_with('{') {
            let block = match arguments_end(rest) {
                Some(end) => rest.split_at(end).0,
                None if at_end => rest,
                None => return Read::Unfinished,
            };
            return match parse_arguments(block) {
                Some(arguments) => Read::Call(WrittenCall { name, arguments }),
                None => Read::NotACall,
            };
        }
        if rest.starts_with('(') {
            let block = match keyword_arguments_end(rest) {
                Some(end) => rest.split_at(end).0,
                None if at_end => rest,
                None => return Read::Unfinished,
            };
            return match parse_keyword_arguments(block) {
                Some(arguments) => Read::Call(WrittenCall { name, arguments }),
                None => Read::NotACall,
            };
        }
        let Some(closer) = closer else {
            return if rest.is_empty() && !at_end {
                Read::Unfinished
            } else {
                Read::NotACall
            };
        };
        if at_end || rest.starts_with(closer) {
            Read::Call(WrittenCall {
                name,
                arguments: Map::new(),
            })
        } else if closer.starts_with(rest) {
            Read::Unfinished
        } else {
            Read::NotACall
        }
    }

    /// `NAME key="value" ... />`. A tag needs `/>` or an attribute to be a call, so a bare
    /// `<NAME>` in prose stays prose.
    fn read_tag(&self, written: &str, mut rest: &str, at_end: bool) -> Read {
        let Some(name) = self.resolve(written) else {
            return Read::NotACall;
        };
        if !rest.starts_with(|c: char| c.is_whitespace() || c == '/') {
            return Read::NotACall;
        }
        let mut arguments = Map::new();
        loop {
            rest = rest.trim_start();
            if rest.is_empty() || rest == "/" {
                return unclosed_tag(name, arguments, at_end);
            }
            if rest.starts_with("/>") || (rest.starts_with('>') && !arguments.is_empty()) {
                return Read::Call(WrittenCall { name, arguments });
            }
            let (key, after_key) = rest.split_at(word_len(rest));
            if key.is_empty() {
                return Read::NotACall;
            }
            let key = key.to_string();
            rest = after_key.trim_start();
            if rest.is_empty() {
                return unclosed_tag(name, arguments, at_end);
            }
            let Some(value) = rest.strip_prefix('=') else {
                return Read::NotACall;
            };
            rest = value.trim_start();
            let Some(quote) = rest.chars().next() else {
                return unclosed_tag(name, arguments, at_end);
            };
            let Some(value) = rest.strip_prefix(['"', '\'']) else {
                return Read::NotACall;
            };
            match value.split_once(quote) {
                Some((value, after)) => {
                    arguments.insert(key, Value::String(value.to_string()));
                    rest = after;
                }
                None => {
                    arguments.insert(key, Value::String(value.to_string()));
                    return unclosed_tag(name, arguments, at_end);
                }
            }
        }
    }

    /// The declared tool a written name means: itself, or the one tool whose name after `__`
    /// matches, since the model drops the extension's prefix.
    fn resolve(&self, written: &str) -> Option<String> {
        if self.tools.iter().any(|tool| tool.name == written) {
            return Some(written.to_string());
        }
        self.resolve_short(written.split_once("__")?.1)
    }

    /// The one declared tool whose name after `__` is `short`.
    fn resolve_short(&self, short: &str) -> Option<String> {
        let mut matches = self.tools.iter().filter(|tool| {
            tool.name
                .split_once("__")
                .is_some_and(|(_, tail)| tail == short)
        });
        let tool = matches.next()?;
        matches.next().is_none().then(|| tool.name.clone())
    }
}

/// The parameter a tool's text alone can fill: its one required parameter when that is a
/// string, or, with none required, its only string parameter.
fn text_parameter(schema: &Map<String, Value>) -> Option<String> {
    let properties = schema.get("properties")?.as_object()?;
    let is_text = |name: &str| {
        properties
            .get(name)
            .and_then(|property| property.get("type"))
            .is_some_and(|kind| match kind {
                Value::String(kind) => kind == "string",
                Value::Array(kinds) => kinds.iter().any(|kind| kind.as_str() == Some("string")),
                _ => false,
            })
    };
    let required: Vec<&str> = schema
        .get("required")
        .and_then(Value::as_array)
        .map(|required| required.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    match required.as_slice() {
        [only] => is_text(only).then(|| only.to_string()),
        [] => {
            let mut texts = properties.keys().filter(|name| is_text(name));
            let only = texts.next()?;
            texts.next().is_none().then(|| only.clone())
        }
        _ => None,
    }
}

/// A tag the text stopped inside: a call once the generation is over, if it has an argument.
fn unclosed_tag(name: String, arguments: Map<String, Value>, at_end: bool) -> Read {
    match (at_end, arguments.is_empty()) {
        (false, _) => Read::Unfinished,
        (true, false) => Read::Call(WrittenCall { name, arguments }),
        (true, true) => Read::NotACall,
    }
}

/// Where a `(key="value",...)` block at the start of `text` closes, just past its `)`, or `None`
/// while it is still open; a quoted string runs to its closing quote.
fn keyword_arguments_end(text: &str) -> Option<usize> {
    let mut depth = 0usize;
    let mut quote = None;
    let mut escaped = false;
    for (at, c) in text.char_indices() {
        match (quote, c) {
            (Some(_), _) if escaped => escaped = false,
            (Some(_), '\\') => escaped = true,
            (Some(open), _) if c == open => quote = None,
            (Some(_), _) => {}
            (None, '"' | '\'') => quote = Some(c),
            (None, '(') => depth += 1,
            (None, ')') => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return Some(at + 1);
                }
            }
            (None, _) => {}
        }
    }
    None
}

/// Reads `(key="value", count=2, flag=True)`, also when the generation cut it short.
fn parse_keyword_arguments(block: &str) -> Option<Map<String, Value>> {
    let mut rest = block.strip_prefix('(')?;
    let mut arguments = Map::new();
    loop {
        rest = rest.trim_start_matches(|c: char| c.is_whitespace() || c == ',');
        if rest.is_empty() || rest.starts_with(')') {
            return Some(arguments);
        }
        let (key, after) = rest.split_at(word_len(rest));
        if key.is_empty() {
            return None;
        }
        let (value, after) =
            keyword_value(after.trim_start().strip_prefix(['=', ':'])?.trim_start());
        arguments.insert(key.to_string(), value);
        rest = after;
    }
}

/// One keyword argument's value, and the text after it.
fn keyword_value(text: &str) -> (Value, &str) {
    if let Some(quote) = text.chars().next().filter(|c| matches!(c, '"' | '\'')) {
        let body = text.get(1..).unwrap_or_default();
        let mut escaped = false;
        let close = body.char_indices().find_map(|(at, c)| match c {
            _ if escaped => {
                escaped = false;
                None
            }
            '\\' => {
                escaped = true;
                None
            }
            _ if c == quote => Some(at),
            _ => None,
        });
        let Some(close) = close else {
            return (Value::String(body.trim_end_matches(')').to_string()), "");
        };
        let (inner, after) = body.split_at(close);
        let value = if quote == '"' {
            serde_json::from_str(text.get(..close + 2).unwrap_or_default())
                .unwrap_or_else(|_| inner.to_string())
        } else {
            inner.replace("\\'", "'")
        };
        return (Value::String(value), after.get(1..).unwrap_or_default());
    }
    let (token, after) = text.split_at(text.find([',', ')']).unwrap_or(text.len()));
    let token = token.trim();
    let value = match token {
        "True" | "true" => Value::Bool(true),
        "False" | "false" => Value::Bool(false),
        "None" | "null" => Value::Null,
        _ => serde_json::from_str::<serde_json::Number>(token)
            .map(Value::Number)
            .unwrap_or_else(|_| Value::String(token.to_string())),
    };
    (value, after)
}

/// The length of a tool or attribute name at the start of `text`.
fn word_len(text: &str) -> usize {
    if !text.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_') {
        return 0;
    }
    text.find(|c: char| !(c.is_ascii_alphanumeric() || c == '_' || c == '-'))
        .unwrap_or(text.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    type Tools = Vec<(&'static str, Map<String, Value>)>;

    fn schema(value: Value) -> Map<String, Value> {
        value.as_object().cloned().unwrap_or_default()
    }

    fn tools() -> Tools {
        vec![
            (
                "giap-knowledge__search_web",
                schema(json!({
                    "type": "object",
                    "properties": {"question": {"type": "string"}, "max_results": {"type": "integer"}},
                    "required": ["question"]
                })),
            ),
            (
                "giap-knowledge__get_wikipedia_article",
                schema(json!({
                    "type": "object",
                    "properties": {"topic": {"type": ["string", "null"]}, "limit": {"type": ["integer", "null"]}}
                })),
            ),
            (
                "giap-system__get_current_time",
                schema(json!({"type": "object", "properties": {}})),
            ),
            (
                "giap-weather__search_web",
                schema(json!({
                    "type": "object",
                    "properties": {"query": {"type": "string"}, "days": {"type": "integer"}},
                    "required": ["query", "days"]
                })),
            ),
        ]
    }

    fn calls_for(tools: &Tools) -> WrittenToolCalls {
        WrittenToolCalls::new(tools.iter().map(|(name, schema)| (*name, schema)))
    }

    /// Feeds `pieces` in order and returns the shown text and the call read.
    fn run_with(tools: &Tools, pieces: &[&str]) -> (String, Option<(String, Value)>) {
        let mut calls = calls_for(tools);
        let mut shown = String::new();
        for piece in pieces {
            shown.push_str(&calls.push(piece));
        }
        shown.push_str(&calls.finish());
        let call = calls
            .take()
            .map(|call| (call.name, Value::Object(call.arguments)));
        (shown, call)
    }

    /// Every split point, and one character at a time, gives what the whole text gives.
    fn read(text: &str) -> (String, Option<(String, Value)>) {
        read_with(&tools(), text)
    }

    fn read_with(tools: &Tools, text: &str) -> (String, Option<(String, Value)>) {
        let whole = run_with(tools, &[text]);
        for (at, _) in text.char_indices().skip(1) {
            let (head, tail) = text.split_at(at);
            assert_eq!(run_with(tools, &[head, tail]), whole, "split at {at}");
        }
        let chars: Vec<String> = text.chars().map(String::from).collect();
        let chars: Vec<&str> = chars.iter().map(String::as_str).collect();
        assert_eq!(run_with(tools, &chars), whole, "one character at a time");
        whole
    }

    fn call(name: &str, arguments: Value) -> Option<(String, Value)> {
        Some((name.to_string(), arguments))
    }

    #[test]
    fn reads_a_self_closing_tag() {
        assert_eq!(
            read(r#"<giap-knowledge__search_web question="what to pack for a day hike" />"#),
            (
                String::new(),
                call(
                    "giap-knowledge__search_web",
                    json!({"question": "what to pack for a day hike"})
                )
            )
        );
    }

    #[test]
    fn reads_a_tag_without_the_extension_prefix() {
        assert_eq!(
            read(r#"<knowledge__get_wikipedia_article topic="why is the sky blue?"/>"#),
            (
                String::new(),
                call(
                    "giap-knowledge__get_wikipedia_article",
                    json!({"topic": "why is the sky blue?"})
                )
            )
        );
    }

    #[test]
    fn reads_a_call_fence() {
        assert_eq!(
            read(
                r#"<call>giap-knowledge__get_wikipedia_article{topic:"crocodile vs alligator"}</call>"#
            ),
            (
                String::new(),
                call(
                    "giap-knowledge__get_wikipedia_article",
                    json!({"topic": "crocodile vs alligator"})
                )
            )
        );
    }

    #[test]
    fn reads_a_call_behind_a_tag_it_made_up() {
        assert_eq!(
            read("<execute_tool>\ngiap-knowledge__search_web{question: \"how to get coffee stain out of cotton shirt\"}\n</execute_tool>"),
            (
                String::new(),
                call(
                    "giap-knowledge__search_web",
                    json!({"question": "how to get coffee stain out of cotton shirt"})
                )
            )
        );
        assert_eq!(
            read("<result>I will search the web for advice on how to fall asleep faster.</result>\n<call>giap-knowledge__search_web{question:<|\"|>how to fall asleep faster<|\"|>}</call>"),
            (
                "<result>I will search the web for advice on how to fall asleep faster.</result>\n"
                    .to_string(),
                call(
                    "giap-knowledge__search_web",
                    json!({"question": "how to fall asleep faster"})
                )
            )
        );
    }

    #[test]
    fn reads_keyword_arguments_and_a_bracketed_call() {
        let text = "I will use the web search tool to find the optimal steeping time for green tea and then answer Jerr.\n<execute_tool>\ngiap-knowledge__search_web(question=\"How long to steep green tea?\")\n</execute_tool>";
        assert_eq!(
            read(text),
            (
                "I will use the web search tool to find the optimal steeping time for green tea and then answer Jerr.\n".to_string(),
                call(
                    "giap-knowledge__search_web",
                    json!({"question": "How long to steep green tea?"})
                )
            )
        );
        assert_eq!(
            read("[knowledge__get_wikipedia_article{topic:<|\"|>onions make you cry when you cut them<|\"|>}]"),
            (
                String::new(),
                call(
                    "giap-knowledge__get_wikipedia_article",
                    json!({"topic": "onions make you cry when you cut them"})
                )
            )
        );
        assert_eq!(
            read("<call>giap-weather__search_web(query='it\\'s (rain)', days=2, hourly=True, note=None)</call>"),
            (
                String::new(),
                call(
                    "giap-weather__search_web",
                    json!({"query": "it's (rain)", "days": 2, "hourly": true, "note": null})
                )
            )
        );
        assert_eq!(
            read("<call>giap-weather__search_web(query=\"rain"),
            (
                String::new(),
                call("giap-weather__search_web", json!({"query": "rain"}))
            )
        );
    }

    #[test]
    fn reads_the_text_of_a_tag_named_for_a_tool() {
        let one_search: Tools = tools()
            .into_iter()
            .filter(|(name, _)| *name != "giap-weather__search_web")
            .collect();
        assert_eq!(
            read_with(
                &one_search,
                "<search_web>What is the story of how the Great Rift Valley formed</search_web>"
            ),
            (
                String::new(),
                call(
                    "giap-knowledge__search_web",
                    json!({"question": "What is the story of how the Great Rift Valley formed"})
                )
            )
        );
        assert_eq!(
            read(
                "Let me look.\n<get_wikipedia_article>\nBread leavening\n</get_wikipedia_article>"
            ),
            (
                "Let me look.\n".to_string(),
                call(
                    "giap-knowledge__get_wikipedia_article",
                    json!({"topic": "Bread leavening"})
                )
            )
        );
        assert_eq!(
            read("<knowledge__get_wikipedia_article>Bread lea</knowled"),
            (
                String::new(),
                call(
                    "giap-knowledge__get_wikipedia_article",
                    json!({"topic": "Bread lea"})
                )
            )
        );
    }

    #[test]
    fn leaves_a_tag_named_for_a_tool_alone_when_its_text_cannot_be_the_call() {
        for text in [
            // `search_web` names two tools.
            "<search_web>rain tomorrow</search_web>",
            // Two required parameters, or none that take text.
            "<giap-weather__search_web>rain</giap-weather__search_web>",
            "<get_current_time>now</get_current_time>",
            // Prose over several lines inside the tag.
            "<get_wikipedia_article>\nBread rises.\nYeast makes gas.\n</get_wikipedia_article>",
            "<get_wikipedia_article></get_wikipedia_article>",
            "<plan>Search the web.</plan>",
        ] {
            assert_eq!(read(text), (text.to_string(), None), "{text}");
        }
    }

    #[test]
    fn finds_the_parameter_text_fills() {
        let parameters: Vec<(&str, Option<String>)> = tools()
            .iter()
            .map(|(name, schema)| (*name, text_parameter(schema)))
            .collect();
        assert_eq!(
            parameters,
            [
                ("giap-knowledge__search_web", Some("question".to_string())),
                (
                    "giap-knowledge__get_wikipedia_article",
                    Some("topic".to_string())
                ),
                ("giap-system__get_current_time", None),
                ("giap-weather__search_web", None),
            ]
        );
    }

    #[test]
    fn reads_a_call_the_generation_cut_short() {
        assert_eq!(
            read(r#"<call>giap-knowledge__get_wikipedia_article{topic:"crocodile vs"#),
            (
                String::new(),
                call(
                    "giap-knowledge__get_wikipedia_article",
                    json!({"topic": "crocodile vs"})
                )
            )
        );
    }

    #[test]
    fn reads_a_tool_call_fence_after_prose() {
        let text = "Sure, let me look that up.\n<tool_call>\ngiap-knowledge__search_web{question:<|\"|>what to pack for a day hike<|\"|>}\n</tool_call>\nHere is what I found";
        assert_eq!(
            read(text),
            (
                "Sure, let me look that up.\n".to_string(),
                call(
                    "giap-knowledge__search_web",
                    json!({"question": "what to pack for a day hike"})
                )
            )
        );
    }

    #[test]
    fn reads_a_tag_at_the_end_of_a_reply() {
        assert_eq!(
            read(
                r#"Bread rises because yeast makes gas. <giap-knowledge__get_wikipedia_article topic="Bread leavening" />"#
            ),
            (
                "Bread rises because yeast makes gas. ".to_string(),
                call(
                    "giap-knowledge__get_wikipedia_article",
                    json!({"topic": "Bread leavening"})
                )
            )
        );
    }

    #[test]
    fn reads_the_native_grammar_and_a_call_without_arguments() {
        assert_eq!(
            read("<tool_call>call:giap-weather__search_web{query:<|\"|>rain<|\"|>,days:2}"),
            (
                String::new(),
                call(
                    "giap-weather__search_web",
                    json!({"query": "rain", "days": 2})
                )
            )
        );
        assert_eq!(
            read("<call>giap-system__get_current_time</call> and more"),
            (
                String::new(),
                call("giap-system__get_current_time", json!({}))
            )
        );
        assert_eq!(
            read("<giap-system__get_current_time/>"),
            (
                String::new(),
                call("giap-system__get_current_time", json!({}))
            )
        );
    }

    #[test]
    fn leaves_undeclared_and_ambiguous_names_alone() {
        for text in [
            r#"<giap-knowledge__read_page url="https://example.com" />"#,
            r#"<search_web question="x" />"#,
            r#"<other__search_web question="x" />"#,
            "<call>giap-knowledge__read_page{url:\"x\"}</call>",
            "<tool_call>\nnot_a_tool{a:1}\n</tool_call>",
        ] {
            assert_eq!(read(text), (text.to_string(), None), "{text}");
        }
    }

    #[test]
    fn leaves_ordinary_markup_and_prose_alone() {
        for text in [
            "<b>bold</b> and <i>x</i>",
            "3 < 5 and 7 > 2",
            "a<b, b<c",
            "Python's __init__ and <__init__>",
            "<giap-knowledge__search_web> is the tool I would use",
            "<giap-knowledge__search_web>",
            "use <call> tags",
            "the <tool_call> fence",
            "ends with <",
            "ends with <giap-knowledge__search_web",
            "<|\"|>quoted<|\"|>",
            "<turn_plan>\n1. Search the web.\n</turn_plan>\nSteep it for one minute.",
            "<response>\nBread rises because of yeast.",
            "See [1] and [the guide](https://example.com/a_b) (page 2).",
            "[giap-knowledge__search_web] is the tool, f(x) = 2",
            "ends with [giap-knowledge__search_web",
        ] {
            assert_eq!(read(text), (text.to_string(), None), "{text}");
        }
    }

    #[test]
    fn streams_prose_before_a_possible_call_at_once() {
        let mut calls = calls_for(&tools());
        assert_eq!(calls.push("Let me check. <giap-know"), "Let me check. ");
        assert_eq!(calls.push("ledge__search_web question=\"ra"), "");
        assert_eq!(calls.push("in\" />and the rest"), "");
        assert!(calls.found());
        assert_eq!(calls.push("more"), "");
        assert_eq!(calls.finish(), "");
        assert_eq!(
            calls.take(),
            Some(WrittenCall {
                name: "giap-knowledge__search_web".to_string(),
                arguments: json!({"question": "rain"}).as_object().unwrap().clone(),
            })
        );
    }

    #[test]
    fn releases_a_tag_as_soon_as_it_cannot_be_a_call() {
        let mut calls = calls_for(&tools());
        assert_eq!(calls.push("x <b"), "x ");
        assert_eq!(calls.push(">bold"), "");
        assert_eq!(calls.push(" text"), "<b>bold text");
        assert_eq!(calls.push(" <giap-knowledge__search_web is"), " ");
        assert_eq!(calls.push(" great"), "<giap-knowledge__search_web is great");
    }

    #[test]
    fn passes_everything_through_without_tools() {
        let mut calls = calls_for(&Vec::new());
        let text = r#"<giap-knowledge__search_web question="x" />"#;
        assert_eq!(calls.push(text), text);
        assert_eq!(calls.finish(), "");
        assert!(!calls.found());
    }

    #[test]
    fn releases_held_text_past_the_cap() {
        let mut calls = calls_for(&tools());
        let open = "<call>giap-knowledge__search_web{question:\"";
        assert_eq!(calls.push(open), "");
        let long = "a".repeat(MAX_HELD_BYTES);
        assert_eq!(calls.push(&long), format!("{open}{long}"));
        assert!(!calls.found());
    }
}
