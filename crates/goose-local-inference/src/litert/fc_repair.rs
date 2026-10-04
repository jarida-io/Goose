//! Reads a Gemma 4 tool call that LiteRT-LM's own parser rejected, so the turn keeps the call
//! instead of failing. A conversation created with `return_error_on_parse_failure` off returns
//! such a call as a text content item with an "error" field.
//!
//! A port of giap-main's `python/litert_lm_cli/commands/serve/fc_repair.py`, which reads a call
//! in the grammar the parser uses (`call:NAME{key:value,...}`, strings between `<|"|>`) and lets
//! a stray string delimiter, an unclosed string, a trailing comma or a missing closing bracket
//! pass. Extended to an unquoted word where a string belongs: gemma-4-E2B wrote
//! `segment:knowledge,tier:permanent`, and the parser failed the whole turn on it.

use std::sync::LazyLock;

use regex::Regex;
use serde_json::{Map, Number, Value};

/// The lexer's string delimiters: each one opens or closes a string.
const ESCAPES: [&str; 3] = ["<|\"|>", "<escape>", "<ctrl46>"];
/// The fences around a Gemma 4 tool call.
const FENCES: [&str; 2] = ["<|tool_call>", "<tool_call|>"];

static IDENTIFIER: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[A-Za-z_][A-Za-z0-9_.\-]*").expect("valid identifier regex"));
static NUMBER: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?$")
        .expect("valid number regex")
});

/// Reads one `call:NAME{...}` block, or `None` when it cannot.
pub(super) fn parse_call(block: &str) -> Option<(String, Map<String, Value>)> {
    let mut text = block.to_string();
    for fence in FENCES {
        text = text.replace(fence, "");
    }
    Reader {
        text: text.trim(),
        pos: 0,
    }
    .call()
    .ok()
}

struct Unreadable;

struct Reader<'a> {
    text: &'a str,
    pos: usize,
}

impl<'a> Reader<'a> {
    fn call(mut self) -> Result<(String, Map<String, Value>), Unreadable> {
        self.skip_space();
        if !self.rest().starts_with("call") {
            return Err(Unreadable);
        }
        self.pos += "call".len();
        self.expect(':')?;
        let name = self.identifier()?;
        self.skip_space();
        if self.at_end() {
            return Ok((name, Map::new()));
        }
        if !self.rest().starts_with('{') {
            return Err(Unreadable);
        }
        Ok((name, self.object()?))
    }

    fn object(&mut self) -> Result<Map<String, Value>, Unreadable> {
        self.pos += 1; // {
        let mut result = Map::new();
        loop {
            self.skip_space();
            if self.at_end() {
                return Ok(result);
            }
            if self.rest().starts_with('}') {
                self.pos += 1;
                return Ok(result);
            }
            if self.rest().starts_with(',') {
                self.pos += 1;
                continue;
            }
            if let Some(escape) = self.escape_here() {
                // A delimiter where a key belongs closes nothing: skip it.
                self.pos += escape.len();
                continue;
            }
            let key = self.identifier()?;
            self.expect(':')?;
            let value = self.value()?;
            result.insert(key, value);
        }
    }

    fn array(&mut self) -> Result<Vec<Value>, Unreadable> {
        self.pos += 1; // [
        let mut items = Vec::new();
        loop {
            self.skip_space();
            if self.at_end() {
                return Ok(items);
            }
            if self.rest().starts_with(']') {
                self.pos += 1;
                return Ok(items);
            }
            if self.rest().starts_with(',') {
                self.pos += 1;
                continue;
            }
            items.push(self.value()?);
        }
    }

    fn value(&mut self) -> Result<Value, Unreadable> {
        self.skip_space();
        if self.at_end() {
            return Err(Unreadable);
        }
        if let Some(escape) = self.escape_here() {
            self.pos += escape.len();
            return Ok(Value::String(self.string()));
        }
        if self.rest().starts_with('{') {
            return Ok(Value::Object(self.object()?));
        }
        if self.rest().starts_with('[') {
            return Ok(Value::Array(self.array()?));
        }
        self.bare()
    }

    /// A string runs to its closing delimiter or, unclosed, to the next comma or closing brace.
    fn string(&mut self) -> String {
        let rest = self.rest();
        let close = ESCAPES
            .iter()
            .filter_map(|escape| rest.find(escape).map(|at| (at, escape.len())))
            .min_by_key(|(at, _)| *at);
        if let Some((at, len)) = close {
            self.pos += at + len;
            return rest.get(..at).unwrap_or_default().to_string();
        }
        let stop = [rest.find(','), rest.find('}')]
            .into_iter()
            .flatten()
            .min()
            .unwrap_or(rest.len());
        self.pos += stop;
        rest.get(..stop).unwrap_or_default().to_string()
    }

    /// An unquoted value, up to the next comma or closing bracket: `true`, `false`, `null`, a
    /// number, or a word meant as a string. Anything not starting like a word or a number is
    /// still unreadable.
    fn bare(&mut self) -> Result<Value, Unreadable> {
        let rest = self.rest();
        let starts_like_a_value = rest
            .chars()
            .next()
            .is_some_and(|first| first.is_alphanumeric() || first == '-');
        if !starts_like_a_value {
            return Err(Unreadable);
        }
        let stop = rest.find([',', '}', ']']).unwrap_or(rest.len());
        let token = rest.get(..stop).unwrap_or_default().trim_end();
        self.pos += stop;
        Ok(match token {
            "true" => Value::Bool(true),
            "false" => Value::Bool(false),
            "null" => Value::Null,
            _ => number(token).unwrap_or_else(|| Value::String(token.to_string())),
        })
    }

    fn identifier(&mut self) -> Result<String, Unreadable> {
        self.skip_space();
        let found = IDENTIFIER.find(self.rest()).ok_or(Unreadable)?;
        let name = found.as_str().to_string();
        self.pos += found.end();
        Ok(name)
    }

    fn expect(&mut self, expected: char) -> Result<(), Unreadable> {
        self.skip_space();
        if !self.rest().starts_with(expected) {
            return Err(Unreadable);
        }
        self.pos += expected.len_utf8();
        Ok(())
    }

    fn escape_here(&self) -> Option<&'static str> {
        ESCAPES
            .iter()
            .copied()
            .find(|escape| self.rest().starts_with(escape))
    }

    fn skip_space(&mut self) {
        let rest = self.rest();
        self.pos += rest.len() - rest.trim_start().len();
    }

    fn at_end(&self) -> bool {
        self.pos >= self.text.len()
    }

    /// What is left to read; borrows the text, not the reader, so `pos` can move under it.
    fn rest(&self) -> &'a str {
        self.text.get(self.pos..).unwrap_or_default()
    }
}

fn number(token: &str) -> Option<Value> {
    if !NUMBER.is_match(token) {
        return None;
    }
    if token.contains(['.', 'e', 'E']) {
        token
            .parse::<f64>()
            .ok()
            .and_then(Number::from_f64)
            .map(Value::Number)
    } else {
        token
            .parse::<i64>()
            .map(|n| Value::Number(n.into()))
            .ok()
            .or_else(|| {
                token
                    .parse::<f64>()
                    .ok()
                    .and_then(Number::from_f64)
                    .map(Value::Number)
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const Q: &str = "<|\"|>";

    fn call(block: &str) -> Option<(String, Value)> {
        parse_call(block).map(|(name, arguments)| (name, Value::Object(arguments)))
    }

    #[test]
    fn reads_a_well_formed_call() {
        assert_eq!(
            call(&format!(
                "call:giap-weather__get_weather_forecast{{days:3,location:{Q}Kisumu{Q}}}"
            )),
            Some((
                "giap-weather__get_weather_forecast".to_string(),
                json!({"days": 3, "location": "Kisumu"})
            ))
        );
    }

    #[test]
    fn skips_a_stray_delimiter() {
        // The call that ended a turn with HTTP 500 on a Jetson Orin.
        assert_eq!(
            call(&format!(
                "<|tool_call>call:giap-draft__list_drafts{{session_id:{Q}{Q}{Q}}}<tool_call|>"
            )),
            Some((
                "giap-draft__list_drafts".to_string(),
                json!({"session_id": ""})
            ))
        );
    }

    #[test]
    fn ends_an_unclosed_string_at_the_next_brace() {
        assert_eq!(
            call(&format!(
                "call:giap-system__send_notification{{title:{Q}Laundry done}}"
            )),
            Some((
                "giap-system__send_notification".to_string(),
                json!({"title": "Laundry done"})
            ))
        );
    }

    #[test]
    fn forgives_a_trailing_comma_and_a_missing_brace() {
        assert_eq!(
            call(&format!(
                "call:giap-finance__convert_currency{{amount:100,from:{Q}USD{Q},to:{Q}KES{Q},"
            )),
            Some((
                "giap-finance__convert_currency".to_string(),
                json!({"amount": 100, "from": "USD", "to": "KES"})
            ))
        );
    }

    #[test]
    fn reads_nested_values() {
        assert_eq!(
            call(&format!("call:t{{a:[1,2.5,true,null],b:{{c:{Q}x, y{Q}}}}}")),
            Some((
                "t".to_string(),
                json!({"a": [1, 2.5, true, null], "b": {"c": "x, y"}})
            ))
        );
    }

    #[test]
    fn reads_a_call_without_arguments() {
        assert_eq!(
            call("call:giap-system__get_current_time"),
            Some(("giap-system__get_current_time".to_string(), json!({})))
        );
    }

    #[test]
    fn reads_a_bare_word_as_a_string() {
        // What gemma-4-E2B wrote, and the engine's parser failed the turn on.
        assert_eq!(
            call(&format!(
                "call:giap-memory__save_memory{{content:{Q}My daughter Wanjiru is seven and \
                 loves drawing.{Q},importance:0.8,segment:knowledge,tier:permanent}}"
            )),
            Some((
                "giap-memory__save_memory".to_string(),
                json!({
                    "content": "My daughter Wanjiru is seven and loves drawing.",
                    "importance": 0.8,
                    "segment": "knowledge",
                    "tier": "permanent"
                })
            ))
        );
        assert_eq!(
            call("call:giap-schedule__set_timer{duration:10 minutes}"),
            Some((
                "giap-schedule__set_timer".to_string(),
                json!({"duration": "10 minutes"})
            ))
        );
    }

    #[test]
    fn gives_up_on_what_is_not_a_call() {
        assert_eq!(call("I could not do that."), None);
        assert_eq!(call("call:{days:3}"), None);
        assert_eq!(call("call:t{days:@}"), None);
    }
}
