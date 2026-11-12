//! The OpenAI-flavor chat-completions backend (`/chat/completions`).
//!
//! One wire format, many gateways: OpenAI itself, OpenRouter, DeepSeek's
//! OpenAI-compatible surface, Groq, Together, xAI - anything that speaks
//! `POST {base}/chat/completions` with `Authorization: Bearer <key>` and
//! streams `data:`-prefixed chunks terminated by `data: [DONE]`.
//!
//! Two differences from [`crate::anthropic`] worth knowing:
//!
//! - **Auth is a bearer token**, not `x-api-key`.
//! - **`base_url` includes the version segment**, following the
//!   `OPENAI_BASE_URL` convention people already have in their heads:
//!   `https://openrouter.ai/api/v1`, `https://api.openai.com/v1`. The
//!   Anthropic backend instead appends `/v1/messages` itself.
//!
//! Some gateways report what they actually billed. OpenRouter puts a `cost`
//! (in USD) on the terminal usage chunk, which this surfaces as
//! [`StreamEvent::Cost`] in cents - a number that cannot drift out of date the
//! way configured [`Pricing`](crate::Pricing) can.

use crate::{SecretString, StreamEvent, TokenUsage, mark_sensitive};
use futures::StreamExt;
use secrecy::ExposeSecret;
use std::collections::VecDeque;

/// A configured chat-completions client. Cheap to construct; holds a reusable
/// `reqwest::Client`. Build with [`Client::new`] and the `with_*` setters.
#[derive(Clone, Debug)]
pub struct Client {
    base_url: String,
    api_key: SecretString,
    model: String,
    max_output_tokens: u32,
    referer: Option<String>,
    title: Option<String>,
    http: reqwest::Client,
    tools: Vec<serde_json::Value>,
}

impl Client {
    /// A client for `model` against `base_url`, which must include the version
    /// segment (e.g. `https://openrouter.ai/api/v1`). A trailing slash is
    /// trimmed. Defaults: `max_output_tokens = 4096`.
    pub fn new(
        base_url: impl Into<String>,
        api_key: SecretString,
        model: impl Into<String>,
    ) -> Self {
        Client {
            base_url: base_url.into().trim_end_matches('/').to_string(),
            api_key,
            model: model.into(),
            max_output_tokens: 4096,
            referer: None,
            title: None,
            http: reqwest::Client::new(),
            tools: Vec::new(),
        }
    }

    /// Cap the answer length. Headroom against runaways, not an editor.
    pub fn with_max_output_tokens(mut self, n: u32) -> Self {
        self.max_output_tokens = n;
        self
    }

    /// OpenRouter attribution headers (`HTTP-Referer`, `X-Title`): they put
    /// your app on OpenRouter's leaderboards and are ignored by every other
    /// gateway. Optional.
    pub fn with_attribution(
        mut self,
        referer: impl Into<String>,
        title: impl Into<String>,
    ) -> Self {
        self.referer = Some(referer.into());
        self.title = Some(title.into());
        self
    }

    /// Reuse a pre-built `reqwest::Client` (connection pool, proxy, timeouts…).
    pub fn with_http_client(mut self, http: reqwest::Client) -> Self {
        self.http = http;
        self
    }

    /// Tool definitions to send to the model. Each is a JSON object in
    /// OpenAI's `tools` shape (`{"type":"function","function":{...}}`).
    pub fn with_tools(mut self, tools: Vec<serde_json::Value>) -> Self {
        self.tools = tools;
        self
    }

    /// The model this client targets.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// The request this client would send. Separate from sending so the
    /// headers can be checked without a network, and so the key is exposed in
    /// exactly one place.
    pub fn build_request(
        &self,
        system: String,
        messages: Vec<serde_json::Value>,
    ) -> reqwest::Result<reqwest::Request> {
        // Unlike Anthropic, there is no top-level `system` field: the system
        // prompt is just the first message.
        let mut msgs = Vec::with_capacity(messages.len() + 1);
        if !system.is_empty() {
            msgs.push(serde_json::json!({"role": "system", "content": system}));
        }
        msgs.extend(messages);

        let mut body = serde_json::json!({
            "model": self.model,
            "max_tokens": self.max_output_tokens,
            "messages": msgs,
            "stream": true,
            // Ask for the figures. OpenAI-compatible providers only report
            // usage on a final, contentless chunk when this is set: without
            // it Reflection returns no usage at all and every call looks free.
            "stream_options": { "include_usage": true },
        });
        if !self.tools.is_empty() {
            body["tools"] = serde_json::json!(self.tools);
        }

        let mut req = self
            .http
            .post(format!("{}/chat/completions", self.base_url))
            .bearer_auth(self.api_key.expose_secret())
            .json(&body);
        if let Some(r) = &self.referer {
            req = req.header("HTTP-Referer", r);
        }
        if let Some(t) = &self.title {
            req = req.header("X-Title", t);
        }
        let mut built = req.build()?;
        mark_sensitive(&mut built, "authorization");
        Ok(built)
    }

    /// One streamed chat completion. `system` is prepended as a `system` role
    /// message, since this wire format has no separate system field. Yields
    /// parsed [`StreamEvent`]s; wire faults surface as [`StreamEvent::Error`].
    pub async fn stream_chat(
        &self,
        system: String,
        messages: Vec<serde_json::Value>,
    ) -> Result<impl futures::Stream<Item = StreamEvent> + Send + 'static, String> {
        let req = self
            .build_request(system, messages)
            .map_err(|e| format!("could not build the request: {e}"))?;
        let resp = self
            .http
            .execute(req)
            .await
            .map_err(|e| format!("upstream request failed: {e}"))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(format!("upstream {status}: {text}"));
        }

        // bytes → (buffered) SSE blocks → parsed events, one at a time.
        let state = (
            resp.bytes_stream(),
            String::new(),
            VecDeque::new(),
            std::collections::HashMap::new(),
        );
        Ok(futures::stream::unfold(
            state,
            |(mut bytes, mut buf, mut queue, mut tool_acc): (
                _,
                _,
                _,
                std::collections::HashMap<usize, (String, String, String)>,
            )| async move {
                loop {
                    if let Some(ev) = queue.pop_front() {
                        return Some((ev, (bytes, buf, queue, tool_acc)));
                    }
                    match bytes.next().await {
                        Some(Ok(chunk)) => {
                            buf.push_str(&String::from_utf8_lossy(&chunk));
                            let events = parse_sse_events(&mut buf, &mut tool_acc);
                            queue.extend(events);
                        }
                        Some(Err(e)) => {
                            queue.push_back(StreamEvent::Error(format!("stream broke: {e}")));
                        }
                        None => return None,
                    }
                }
            },
        ))
    }
}

/// Consume every COMPLETE server-sent event in `buf`, returning what they parse
/// to and leaving any trailing partial event buffered for the next chunk. Pure,
/// so the wire handling is unit-testable without a socket.
///
/// `data: [DONE]` is this format's end-of-stream marker and maps to
/// [`StreamEvent::Done`] - unlike Anthropic, which sends a typed
/// `message_stop`.
pub fn parse_sse_events(
    buf: &mut String,
    tool_acc: &mut std::collections::HashMap<usize, (String, String, String)>,
) -> Vec<StreamEvent> {
    let mut out = Vec::new();
    if buf.contains("\r\n") {
        *buf = buf.replace("\r\n", "\n");
    }
    while let Some(end) = buf.find("\n\n") {
        let block: String = buf.drain(..end + 2).collect();
        for line in block.lines() {
            let Some(data) = line.strip_prefix("data:") else {
                continue;
            };
            let data = data.trim();
            if data.is_empty() {
                continue;
            }
            if data == "[DONE]" {
                out.push(StreamEvent::Done);
                continue;
            }
            let Ok(v) = serde_json::from_str::<serde_json::Value>(data) else {
                continue;
            };

            // A gateway may report an error mid-stream rather than by status.
            if let Some(msg) = v
                .pointer("/error/message")
                .and_then(|m| m.as_str())
                .or_else(|| v.pointer("/error").and_then(|m| m.as_str()))
            {
                out.push(StreamEvent::Error(msg.to_string()));
                continue;
            }

            if let Some(text) = v
                .pointer("/choices/0/delta/content")
                .and_then(|t| t.as_str())
                .filter(|s| !s.is_empty())
            {
                out.push(StreamEvent::TextDelta(text.to_string()));
            }
            // Reasoning content (Beam requires this in multi-turn)
            for key in ["reasoning_content", "reasoning"] {
                if let Some(reasoning) = v
                    .pointer(&format!("/choices/0/delta/{}", key))
                    .and_then(|t| t.as_str())
                    .filter(|s| !s.is_empty())
                {
                    out.push(StreamEvent::ReasoningDelta(reasoning.to_string()));
                    break;
                }
            }
            // Tool calls
            if let Some(tc_arr) = v
                .pointer("/choices/0/delta/tool_calls")
                .and_then(|t| t.as_array())
            {
                for tc in tc_arr {
                    let idx = tc.get("index").and_then(|i| i.as_u64()).unwrap_or(0) as usize;
                    let entry = tool_acc
                        .entry(idx)
                        .or_insert_with(|| (String::new(), String::new(), String::new()));
                    if let Some(id) = tc.get("id").and_then(|i| i.as_str()) {
                        if !id.is_empty() {
                            entry.0 = id.to_string();
                        }
                    }
                    if let Some(name) = tc.pointer("/function/name").and_then(|n| n.as_str()) {
                        if !name.is_empty() {
                            entry.1 = name.to_string();
                        }
                    }
                    if let Some(args) = tc.pointer("/function/arguments").and_then(|a| a.as_str()) {
                        entry.2.push_str(args);
                    }
                }
            }
            if let Some(reason) = v
                .pointer("/choices/0/finish_reason")
                .and_then(|r| r.as_str())
            {
                if reason == "tool_calls" {
                    // Emit accumulated tool calls
                    let mut indices: Vec<usize> = tool_acc.keys().copied().collect();
                    indices.sort();
                    for idx in indices {
                        if let Some((id, name, args)) = tool_acc.remove(&idx) {
                            out.push(StreamEvent::ToolCall {
                                id,
                                name,
                                arguments: args,
                            });
                        }
                    }
                }
                out.push(StreamEvent::StopReason(normalize_finish_reason(reason)));
            }
            if let Some(usage) = v.get("usage").filter(|u| u.is_object()) {
                out.push(StreamEvent::Usage(parse_usage(usage)));
                // OpenRouter reports what it actually billed, in USD.
                if let Some(usd) = usage.get("cost").and_then(|c| c.as_f64()) {
                    out.push(StreamEvent::Cost(usd * 100.0));
                }
            }
        }
    }
    out
}

/// Map this format's `finish_reason` onto the vocabulary callers already match
/// on, which is Anthropic's. Only `length` needs translating: it is the
/// token-cap cutoff that Anthropic calls `max_tokens`, and callers key their
/// "the answer was truncated" handling off that exact string.
fn normalize_finish_reason(reason: &str) -> String {
    match reason {
        "length" => "max_tokens".to_string(),
        other => other.to_string(),
    }
}

/// Project an OpenAI-flavor `usage` object onto [`TokenUsage`].
///
/// The subtraction matters: in this format `prompt_tokens` is the TOTAL input
/// including anything served from cache, whereas `TokenUsage::input_tokens` is
/// the uncached remainder that bills at the full input rate. Passing
/// `prompt_tokens` through whole would double-count every cached token against
/// [`crate::usage_cost_cents`]. Saturating, because a gateway reporting details
/// that exceed the total should not wrap.
fn parse_usage(u: &serde_json::Value) -> TokenUsage {
    let g = |p: &str| u.pointer(p).and_then(|n| n.as_u64()).unwrap_or(0);
    let prompt = g("/prompt_tokens");
    // Two spellings of the same number: OpenAI nests it under
    // prompt_tokens_details, DeepSeek reports `prompt_cache_hit_tokens` at the
    // top level and charges a fraction of the miss rate for it. Ignoring the
    // DeepSeek field counted every cached token as ordinary input, so cost
    // tracking ran up to ~50x high and the cache rate was invisible.
    let cached = u
        .pointer("/prompt_tokens_details/cached_tokens")
        .and_then(|n| n.as_u64())
        .or_else(|| {
            u.pointer("/prompt_cache_hit_tokens")
                .and_then(|n| n.as_u64())
        })
        .unwrap_or(0);
    let cache_write = g("/prompt_tokens_details/cache_write_tokens");
    TokenUsage {
        input_tokens: prompt.saturating_sub(cached).saturating_sub(cache_write),
        output_tokens: g("/completion_tokens"),
        cache_creation_tokens: cache_write,
        cache_read_tokens: cached,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_deltas_across_split_chunks_and_ends_on_done() {
        let mut buf = String::new();
        buf.push_str(
            "data: {\"choices\":[{\"delta\":{\"content\":\"The party \"}}]}\n\n\
             data: {\"choices\":[{\"delta\":{\"content\":\"fl",
        );
        assert_eq!(
            parse_sse_events(&mut buf, &mut std::collections::HashMap::new()),
            vec![StreamEvent::TextDelta("The party ".into())]
        );
        assert!(buf.starts_with("data:"), "partial event stays buffered");

        buf.push_str("ed.\"}}]}\n\ndata: [DONE]\n\n");
        assert_eq!(
            parse_sse_events(&mut buf, &mut std::collections::HashMap::new()),
            vec![StreamEvent::TextDelta("fled.".into()), StreamEvent::Done]
        );
        assert!(buf.is_empty());
    }

    #[test]
    fn openrouter_usage_chunk_yields_tokens_and_reported_cost() {
        // The shape OpenRouter documents; usage rides the terminal chunk and is
        // always present now (its `include_usage` opt-ins are deprecated no-ops).
        let mut buf = "data: {\"object\":\"chat.completion.chunk\",\"usage\":{\
             \"completion_tokens\":2,\"cost\":0.95,\
             \"prompt_tokens\":194,\
             \"prompt_tokens_details\":{\"cached_tokens\":0,\"cache_write_tokens\":100}}}\n\n"
            .to_string();
        assert_eq!(
            parse_sse_events(&mut buf, &mut std::collections::HashMap::new()),
            vec![
                StreamEvent::Usage(TokenUsage {
                    // 194 total prompt, of which 100 were cache writes.
                    input_tokens: 94,
                    output_tokens: 2,
                    cache_creation_tokens: 100,
                    cache_read_tokens: 0,
                }),
                // $0.95 reported → 95 cents.
                StreamEvent::Cost(95.0),
            ],
        );
    }

    #[test]
    fn cached_tokens_are_not_double_counted_as_input() {
        // prompt_tokens includes cached reads; input_tokens must be the
        // uncached remainder or cost accounting bills them twice.
        let mut buf = "data: {\"usage\":{\"prompt_tokens\":1000,\"completion_tokens\":10,\
             \"prompt_tokens_details\":{\"cached_tokens\":900}}}\n\n"
            .to_string();
        let events = parse_sse_events(&mut buf, &mut std::collections::HashMap::new());
        let StreamEvent::Usage(u) = &events[0] else {
            panic!("expected usage, got {events:?}");
        };
        assert_eq!(u.input_tokens, 100);
        assert_eq!(u.cache_read_tokens, 900);
        assert_eq!(u.input_tokens + u.cache_read_tokens, 1000);
    }

    #[test]
    fn the_request_asks_for_usage_and_a_contentless_chunk_still_counts() {
        // Reflection reports no usage at all unless stream_options asks, and
        // then only on a final chunk whose choices array is empty.
        let c = Client::new("https://example.invalid/v1", SecretString::from("***"), "m");
        let req = c
            .build_request(String::new(), vec![])
            .expect("request builds");
        let body: serde_json::Value = serde_json::from_slice(
            req.body().and_then(|b| b.as_bytes()).expect("json body"),
        )
        .expect("body is json");
        assert_eq!(body["stream_options"]["include_usage"], true);
        assert_eq!(body["stream"], true);

        let chunk = serde_json::json!({
            "choices": [],
            "usage": {"prompt_tokens": 48, "completion_tokens": 36,
                      "prompt_tokens_details": {"cached_tokens": 0}}
        });
        let usage = parse_usage(&chunk["usage"]);
        assert_eq!(usage.input_tokens, 48);
        assert_eq!(usage.output_tokens, 36);
        // An empty choices array must not be an error: every pointer read
        // from it simply misses.
        assert!(chunk.pointer("/choices/0/delta/content").is_none());
    }

    #[test]
    fn deepseek_cache_hits_are_read_from_the_flat_field() {
        // DeepSeek reports the same number without the nested object:
        // {"prompt_cache_hit_tokens": 900, "prompt_cache_miss_tokens": 100}.
        // Ignoring it billed every cached token at the miss rate.
        let mut buf = "data: {\"usage\":{\"prompt_tokens\":1000,\"completion_tokens\":10,\
             \"prompt_cache_hit_tokens\":900,\"prompt_cache_miss_tokens\":100}}\n\n"
            .to_string();
        let mut acc = std::collections::HashMap::new();
        let events = parse_sse_events(&mut buf, &mut acc);
        let StreamEvent::Usage(u) = &events[0] else {
            panic!("expected usage, got {events:?}");
        };
        assert_eq!(u.cache_read_tokens, 900);
        assert_eq!(u.input_tokens, 100);
        assert_eq!(u.input_tokens + u.cache_read_tokens, 1000);
    }

    #[test]
    fn the_nested_field_still_wins_when_both_are_present() {
        let mut buf = "data: {\"usage\":{\"prompt_tokens\":1000,\"completion_tokens\":10,\
             \"prompt_tokens_details\":{\"cached_tokens\":800},\
             \"prompt_cache_hit_tokens\":900}}\n\n"
            .to_string();
        let mut acc = std::collections::HashMap::new();
        let events = parse_sse_events(&mut buf, &mut acc);
        let StreamEvent::Usage(u) = &events[0] else {
            panic!("expected usage, got {events:?}");
        };
        assert_eq!(u.cache_read_tokens, 800);
    }

    #[test]
    fn length_finish_reason_maps_to_the_anthropic_vocabulary() {
        // Callers key truncation off the literal "max_tokens"; this format says
        // "length" for the same thing.
        let mut buf =
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}]}\n\n".to_string();
        assert_eq!(
            parse_sse_events(&mut buf, &mut std::collections::HashMap::new()),
            vec![StreamEvent::StopReason("max_tokens".into())]
        );

        let mut buf =
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n".to_string();
        assert_eq!(
            parse_sse_events(&mut buf, &mut std::collections::HashMap::new()),
            vec![StreamEvent::StopReason("stop".into())]
        );
    }

    #[test]
    fn real_openrouter_terminal_chunks_captured_from_the_wire() {
        // Captured live from OpenRouter (deepseek-v4-flash-0731, 2026-09-05),
        // ids trimmed. Three things a hand-written fixture would have missed:
        //
        //  1. `finish_reason` arrives TWICE - once on its own chunk, then again
        //     on the terminal chunk that carries usage. Both are reported; the
        //     wire really does say it twice, and swallowing one would mean
        //     guessing which.
        //  2. `delta.reasoning` carries chain-of-thought on some models. It is
        //     not `content`, so it never reaches the answer - matching the
        //     Anthropic backend, which likewise drops thinking deltas.
        //  3. `completion_tokens` INCLUDES `reasoning_tokens`, which is correct
        //     for billing: they bill at the output rate.
        let mut buf = concat!(
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"\",\"role\":\"assistant\",",
            "\"reasoning\":\" colors\"},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Red\",\"role\":\"assistant\"},",
            "\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"\",\"role\":\"assistant\",",
            "\"reasoning\":null},\"finish_reason\":\"stop\",\"native_finish_reason\":\"stop\"}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"\",\"role\":\"assistant\"},",
            "\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":87,\"completion_tokens\":19,",
            "\"total_tokens\":106,\"cost\":0.0000062475,\"is_byok\":false,",
            "\"prompt_tokens_details\":{\"cached_tokens\":0,\"cache_write_tokens\":0},",
            "\"completion_tokens_details\":{\"reasoning_tokens\":9}}}\n\n",
            "data: [DONE]\n\n",
        )
        .to_string();

        let events = parse_sse_events(&mut buf, &mut std::collections::HashMap::new());
        assert_eq!(
            events,
            vec![
                // The reasoning-only chunk contributes no answer text.
                StreamEvent::ReasoningDelta(" colors".into()),
                StreamEvent::TextDelta("Red".into()),
                StreamEvent::StopReason("stop".into()),
                StreamEvent::StopReason("stop".into()),
                StreamEvent::Usage(TokenUsage {
                    input_tokens: 87,
                    output_tokens: 19, // 10 answer + 9 reasoning, both billed
                    cache_creation_tokens: 0,
                    cache_read_tokens: 0,
                }),
                // $0.0000062475 → 0.00062475 cents. Sub-cent by three orders of
                // magnitude, which is what makes the caller's 1c floor the
                // dominant term for short questions on a model this cheap.
                StreamEvent::Cost(0.00062475),
                StreamEvent::Done,
            ],
        );
        assert!(buf.is_empty());
    }

    #[test]
    fn mid_stream_errors_surface() {
        let mut buf =
            "data: {\"error\":{\"message\":\"rate limited\",\"code\":429}}\n\n".to_string();
        assert_eq!(
            parse_sse_events(&mut buf, &mut std::collections::HashMap::new()),
            vec![StreamEvent::Error("rate limited".into())]
        );
    }

    #[test]
    fn empty_role_priming_delta_emits_nothing() {
        // The first chunk usually carries {"role":"assistant"} and no content.
        let mut buf =
            "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"\"}}]}\n\n"
                .to_string();
        assert!(parse_sse_events(&mut buf, &mut std::collections::HashMap::new()).is_empty());
    }

    #[test]
    fn one_call_arguments_split_across_chunks() {
        // A single tool call whose arguments arrive in three pieces. The call
        // is emitted whole once finish_reason says tool_calls.
        let mut acc = std::collections::HashMap::new();
        let mut buf = "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c1\",\"type\":\"function\",\"function\":{\"name\":\"read_file\",\"arguments\":\"{\\\"pa\"}}]}}]}\n\n".to_string();
        assert!(parse_sse_events(&mut buf, &mut acc).is_empty());

        buf.push_str("data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"th\\\": \\\"a\"}}]}}]}\n\n");
        assert!(parse_sse_events(&mut buf, &mut acc).is_empty());

        buf.push_str("data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\".txt\\\"}\"}}]}}]}\n\n");
        assert!(
            parse_sse_events(&mut buf, &mut acc).is_empty(),
            "no call until finish_reason"
        );

        buf.push_str("data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n");
        assert_eq!(
            parse_sse_events(&mut buf, &mut acc),
            vec![
                StreamEvent::ToolCall {
                    id: "c1".into(),
                    name: "read_file".into(),
                    arguments: "{\"path\": \"a.txt\"}".into(),
                },
                StreamEvent::StopReason("tool_calls".into()),
            ]
        );
        assert!(acc.is_empty(), "accumulator drained");
    }

    #[test]
    fn two_parallel_calls_with_interleaved_indices() {
        // Two calls, their argument chunks interleaved. Each must keep its own
        // id, name and arguments; index is what tells them apart.
        let mut acc = std::collections::HashMap::new();
        let mut buf = String::new();
        buf.push_str("data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"a\",\"type\":\"function\",\"function\":{\"name\":\"list_dir\",\"arguments\":\"{\\\"pa\"}},{\"index\":1,\"id\":\"b\",\"type\":\"function\",\"function\":{\"name\":\"read_file\",\"arguments\":\"{\\\"path\\\":\"}}]}}]}\n\n");
        assert!(parse_sse_events(&mut buf, &mut acc).is_empty());

        buf.push_str("data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"th\\\": \\\".\\\"}\"}},{\"index\":1,\"function\":{\"arguments\":\"\\\"f.txt\\\"}\"}}]}}]}\n\n");
        assert!(parse_sse_events(&mut buf, &mut acc).is_empty());

        buf.push_str("data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n");
        assert_eq!(
            parse_sse_events(&mut buf, &mut acc),
            vec![
                StreamEvent::ToolCall {
                    id: "a".into(),
                    name: "list_dir".into(),
                    arguments: "{\"path\": \".\"}".into(),
                },
                StreamEvent::ToolCall {
                    id: "b".into(),
                    name: "read_file".into(),
                    arguments: "{\"path\":\"f.txt\"}".into(),
                },
                StreamEvent::StopReason("tool_calls".into()),
            ]
        );
    }

    #[test]
    fn text_then_a_call_in_one_response() {
        // Prose first, then a call: both come out, in order, and the text is
        // not swallowed by the call handling.
        let mut acc = std::collections::HashMap::new();
        let mut buf =
            "data: {\"choices\":[{\"delta\":{\"content\":\"Let me look. \"}}]}\n\n".to_string();
        assert_eq!(
            parse_sse_events(&mut buf, &mut acc),
            vec![StreamEvent::TextDelta("Let me look. ".into())]
        );

        buf.push_str("data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c1\",\"type\":\"function\",\"function\":{\"name\":\"list_dir\",\"arguments\":\"{\\\"path\\\":\\\".\\\"}\"}}]}}]}\n\n");
        assert!(parse_sse_events(&mut buf, &mut acc).is_empty());

        buf.push_str("data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\ndata: [DONE]\n\n");
        assert_eq!(
            parse_sse_events(&mut buf, &mut acc),
            vec![
                StreamEvent::ToolCall {
                    id: "c1".into(),
                    name: "list_dir".into(),
                    arguments: "{\"path\":\".\"}".into(),
                },
                StreamEvent::StopReason("tool_calls".into()),
                StreamEvent::Done,
            ]
        );
    }

    #[test]
    fn debug_never_prints_the_key() {
        let c = Client::new(
            "https://example.invalid/v1",
            SecretString::from("sk-SECRET-key"),
            "m",
        );
        let s = format!("{c:?}");
        assert!(!s.contains("SECRET"), "Debug printed the key: {s}");
    }

    #[test]
    fn the_key_reaches_the_auth_header_marked_sensitive() {
        let c = Client::new(
            "https://example.invalid/v1",
            SecretString::from("sk-SECRET-key"),
            "m",
        );
        let req = c
            .build_request(String::new(), vec![])
            .expect("request builds");
        let auth = req
            .headers()
            .get("authorization")
            .expect("authorization header");
        assert_eq!(auth.to_str().unwrap(), "Bearer sk-SECRET-key");
        assert!(auth.is_sensitive());
    }
}
