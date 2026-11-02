//! A tool call the model wrote as text instead of in its native wrapper. llama.cpp's parser
//! streams it as content, so the call never runs; Gemma 4 does this with thinking off, as
//! `<NAME key="value" />` or as its own `NAME{key:value}` grammar behind `<call>` or
//! `<tool_call>`. This reads such a call out of the visible content when it names a declared
//! tool, and holds back only text that could still become one.

use serde_json::{Map, Value};

use crate::litert::fc_repair::{arguments_end, parse_arguments};

/// Fences the model put around a call in its own grammar instead of `<|tool_call>`.
const FENCES: [&str; 2] = ["call>", "tool_call>"];
const FENCE_CLOSERS: [&str; 2] = ["</call>", "</tool_call>"];
/// Held text past this is released: no written call has come near it.
const MAX_HELD_BYTES: usize = 4096;

#[derive(Debug, PartialEq)]
pub(super) struct WrittenCall {
    pub name: String,
    pub arguments: Map<String, Value>,
}

pub(super) struct WrittenToolCalls {
    tools: Vec<String>,
    held: String,
    call: Option<WrittenCall>,
}

enum Read {
    Call(WrittenCall),
    Unfinished,
    NotACall,
}

impl WrittenToolCalls {
    pub(super) fn new<'a>(declared: impl IntoIterator<Item = &'a str>) -> Self {
        Self {
            tools: declared.into_iter().map(str::to_string).collect(),
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
        while let Some(offset) = self.held.get(from..).and_then(|rest| rest.find('<')) {
            let at = from + offset;
            match self.read(self.held.get(at + 1..).unwrap_or_default(), at_end) {
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
        for fence in FENCES {
            if let Some(body) = text.strip_prefix(fence) {
                return self.read_fenced(body, at_end);
            }
            if fence.starts_with(text) && !at_end {
                return Read::Unfinished;
            }
        }
        self.read_tag(text, at_end)
    }

    /// `NAME{key:value,...}`, optionally as `call:NAME{...}`, after a fence.
    fn read_fenced(&self, body: &str, at_end: bool) -> Read {
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
        if at_end || FENCE_CLOSERS.iter().any(|closer| rest.starts_with(closer)) {
            Read::Call(WrittenCall {
                name,
                arguments: Map::new(),
            })
        } else if FENCE_CLOSERS.iter().any(|closer| closer.starts_with(rest)) {
            Read::Unfinished
        } else {
            Read::NotACall
        }
    }

    /// `NAME key="value" ... />`. A tag needs `/>` or an attribute to be a call, so a bare
    /// `<NAME>` in prose stays prose.
    fn read_tag(&self, text: &str, at_end: bool) -> Read {
        let (written, mut rest) = text.split_at(word_len(text));
        if written.is_empty() {
            return Read::NotACall;
        }
        if rest.is_empty() {
            return if at_end {
                Read::NotACall
            } else {
                Read::Unfinished
            };
        }
        let Some(name) = self.resolve(written) else {
            return Read::NotACall;
        };
        if !rest.starts_with(|c: char| c.is_whitespace() || c == '/' || c == '>') {
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
        if self.tools.iter().any(|tool| tool == written) {
            return Some(written.to_string());
        }
        let (_, short) = written.split_once("__")?;
        let mut matches = self
            .tools
            .iter()
            .filter(|tool| tool.split_once("__").is_some_and(|(_, tail)| tail == short));
        let tool = matches.next()?;
        matches.next().is_none().then(|| tool.clone())
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

    const TOOLS: [&str; 4] = [
        "giap-knowledge__search_web",
        "giap-knowledge__get_wikipedia_article",
        "giap-system__get_current_time",
        "giap-weather__search_web",
    ];

    /// Feeds `pieces` in order and returns the shown text and the call read.
    fn run(pieces: &[&str]) -> (String, Option<(String, Value)>) {
        let mut calls = WrittenToolCalls::new(TOOLS);
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
        let whole = run(&[text]);
        for (at, _) in text.char_indices().skip(1) {
            let (head, tail) = text.split_at(at);
            assert_eq!(run(&[head, tail]), whole, "split at {at}");
        }
        let chars: Vec<String> = text.chars().map(String::from).collect();
        let chars: Vec<&str> = chars.iter().map(String::as_str).collect();
        assert_eq!(run(&chars), whole, "one character at a time");
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
        ] {
            assert_eq!(read(text), (text.to_string(), None), "{text}");
        }
    }

    #[test]
    fn streams_prose_before_a_possible_call_at_once() {
        let mut calls = WrittenToolCalls::new(TOOLS);
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
        let mut calls = WrittenToolCalls::new(TOOLS);
        assert_eq!(calls.push("x <b"), "x ");
        assert_eq!(calls.push(">bold"), "<b>bold");
        assert_eq!(calls.push(" <giap-knowledge__search_web is"), " ");
        assert_eq!(calls.push(" great"), "<giap-knowledge__search_web is great");
    }

    #[test]
    fn passes_everything_through_without_tools() {
        let mut calls = WrittenToolCalls::new([]);
        let text = r#"<giap-knowledge__search_web question="x" />"#;
        assert_eq!(calls.push(text), text);
        assert_eq!(calls.finish(), "");
        assert!(!calls.found());
    }

    #[test]
    fn releases_held_text_past_the_cap() {
        let mut calls = WrittenToolCalls::new(TOOLS);
        let open = "<call>giap-knowledge__search_web{question:\"";
        assert_eq!(calls.push(open), "");
        let long = "a".repeat(MAX_HELD_BYTES);
        assert_eq!(calls.push(&long), format!("{open}{long}"));
        assert!(!calls.found());
    }
}
