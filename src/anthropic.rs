//! The Anthropic Messages API backend (`/v1/messages`).
//!
//! Endpoint-agnostic within the Anthropic wire format: point [`Client`] at
//! Anthropic directly, or at any compatible gateway, DeepSeek documents
//! `base_url = https://api.deepseek.com/anthropic`, and Grok exposes a similar
//! compat surface. Yields the crate's neutral [`StreamEvent`](crate::StreamEvent)s.

use crate::{StreamEvent, TokenUsage};
use futures::StreamExt;
use std::collections::VecDeque;

pub mod thinking;
pub use thinking::{Effort, ThinkingControl, supported_effort, thinking_control};

/// [`crate::Pricing`] for Claude Haiku 4.5: $1/MTok in, $5/MTok out, cache
/// write 1.25× / read 0.1×. A convenience default; real deployments usually
/// read prices from config.
pub const HAIKU_4_5: crate::Pricing = crate::Pricing {
    input_cents_per_mtok: 100.0,
    output_cents_per_mtok: 500.0,
    cache_write_cents_per_mtok: Some(125.0),
    cache_read_cents_per_mtok: Some(10.0),
};

/// A configured Messages API client. Cheap to construct; holds a reusable
/// `reqwest::Client`. Build with [`Client::new`] and the `with_*` setters.
#[derive(Clone)]
pub struct Client {
    base_url: String,
    api_key: String,
    model: String,
    max_output_tokens: u32,
    adaptive_thinking: bool,
    thinking_budget_tokens: Option<u32>,
    effort: Option<Effort>,
    http: reqwest::Client,
}

impl Client {
    /// A client for `model`, talking to Anthropic directly. Defaults:
    /// `base_url = https://api.anthropic.com`, `max_output_tokens = 4096`,
    /// no `thinking` or `effort` sent (the model's own defaults apply: Opus 5,
    /// Opus 5.5, Sonnet 5 and Fable think by default, older models don't).
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Client {
            base_url: "https://api.anthropic.com".into(),
            api_key: api_key.into(),
            model: model.into(),
            max_output_tokens: 4096,
            adaptive_thinking: false,
            thinking_budget_tokens: None,
            effort: None,
            http: reqwest::Client::new(),
        }
    }

    /// Point at a compatible gateway (e.g. `https://api.deepseek.com/anthropic`).
    /// A trailing slash is trimmed.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into().trim_end_matches('/').to_string();
        self
    }

    /// Cap the answer length. This is headroom against runaways, not an editor -
    /// the system prompt shapes real length. On a model that thinks
    /// adaptively (and Opus 5, Opus 5.5, Sonnet 5 and Fable do by default)
    /// thinking counts toward this cap even though its text never streams, so
    /// size it for the thinking as well as the answer, or pass a
    /// [`with_thinking_budget`](Self::with_thinking_budget) as extra headroom.
    pub fn with_max_output_tokens(mut self, n: u32) -> Self {
        self.max_output_tokens = n;
        self
    }

    /// Turn thinking on and let the model decide how much:
    /// `thinking: {type: "adaptive"}`, on models that support it (Opus 4.6,
    /// Sonnet 4.6 and later; see [`thinking_control`]). Steer depth with
    /// [`with_effort`](Self::with_effort). Models that only take a fixed
    /// budget (Haiku 4.5 and older) get no thinking from this alone; give them
    /// one with [`with_thinking_budget`](Self::with_thinking_budget).
    ///
    /// Thinking deltas are parsed and dropped; only answer text streams out.
    /// Thinking bills at the output rate and adds latency.
    pub fn with_adaptive_thinking(mut self) -> Self {
        self.adaptive_thinking = true;
        self
    }

    /// Turn thinking on with `budget` tokens for it. What goes on the wire
    /// depends on the model ([`thinking_control`]):
    ///
    /// - Models that take only a fixed budget (Haiku 4.5, Sonnet/Opus 4.5 and
    ///   older, non-Claude models such as DeepSeek's compat endpoint) get
    ///   `thinking: {type: "enabled", budget_tokens: budget}`.
    /// - Models that take adaptive thinking (Opus 4.6 / Sonnet 4.6 and later)
    ///   get `thinking: {type: "adaptive"}` instead. `budget_tokens` is a 400
    ///   on Opus 4.7 and later, Sonnet 5 and Fable, and deprecated on 4.6, so
    ///   it is never sent to them.
    ///
    /// Either way `budget` is added to `max_tokens`, so the answer keeps its
    /// own [`with_max_output_tokens`](Self::with_max_output_tokens) headroom
    /// on top of the thinking.
    pub fn with_thinking_budget(mut self, budget: u32) -> Self {
        self.thinking_budget_tokens = Some(budget);
        self
    }

    /// Set `output_config: {effort}`, the control for how much the model
    /// thinks and writes. Sent at the nearest level the model supports, and
    /// not at all to models that reject the parameter (Haiku 4.5, Sonnet 4.5
    /// and older, and non-Claude models); see [`supported_effort`]. Effort is
    /// independent of thinking: on Opus 4.7/4.8 it applies with thinking off,
    /// and on Opus 5.5, whose thinking can't be turned off (its default
    /// effort is `medium`), it is the only lever.
    pub fn with_effort(mut self, effort: Effort) -> Self {
        self.effort = Some(effort);
        self
    }

    /// Reuse a pre-built `reqwest::Client` (connection pool, proxy, timeouts…).
    pub fn with_http_client(mut self, http: reqwest::Client) -> Self {
        self.http = http;
        self
    }

    /// The model this client targets.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// The request body [`stream_chat`](Self::stream_chat) sends, with the
    /// thinking and effort settings resolved for this model. Public so the
    /// shape can be inspected and tested without a network call.
    pub fn request_body(
        &self,
        system: String,
        messages: Vec<serde_json::Value>,
    ) -> serde_json::Value {
        let mut body = serde_json::json!({
            "model": self.model,
            "max_tokens": self.max_output_tokens,
            "system": system,
            "messages": messages,
            "stream": true,
        });
        let wants_thinking = self.adaptive_thinking || self.thinking_budget_tokens.is_some();
        if wants_thinking {
            match thinking_control(&self.model) {
                ThinkingControl::Budget => {
                    if let Some(budget) = self.thinking_budget_tokens {
                        body["thinking"] = serde_json::json!({
                            "type": "enabled",
                            "budget_tokens": budget,
                        });
                    }
                }
                ThinkingControl::AdaptivePreferred | ThinkingControl::AdaptiveOnly => {
                    body["thinking"] = serde_json::json!({ "type": "adaptive" });
                }
            }
            // A budget is thinking headroom on top of the answer's cap. On the
            // budget form Anthropic requires budget_tokens < max_tokens; on
            // adaptive, thinking counts toward max_tokens all the same.
            if let (Some(_), Some(budget)) = (body.get("thinking"), self.thinking_budget_tokens) {
                body["max_tokens"] = self.max_output_tokens.saturating_add(budget).into();
            }
        }
        if let Some(effort) = self.effort.and_then(|e| supported_effort(&self.model, e)) {
            body["output_config"] = serde_json::json!({ "effort": effort.as_str() });
        }
        body
    }

    /// One streamed chat completion. `messages` are `{"role", "content"}` JSON
    /// objects, most-recent last; `system` is the system prompt. Yields parsed
    /// [`StreamEvent`]s; wire faults surface as [`StreamEvent::Error`] so
    /// callers handle exactly one shape.
    pub async fn stream_chat(
        &self,
        system: String,
        messages: Vec<serde_json::Value>,
    ) -> Result<impl futures::Stream<Item = StreamEvent> + Send + 'static, String> {
        let body = self.request_body(system, messages);
        let resp = self
            .http
            .post(format!("{}/v1/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("upstream request failed: {e}"))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(format!("upstream {status}: {text}"));
        }

        // bytes → (buffered) SSE blocks → parsed events, one at a time.
        let state = (resp.bytes_stream(), String::new(), VecDeque::new());
        Ok(futures::stream::unfold(
            state,
            |(mut bytes, mut buf, mut queue)| async move {
                loop {
                    if let Some(ev) = queue.pop_front() {
                        return Some((ev, (bytes, buf, queue)));
                    }
                    match bytes.next().await {
                        Some(Ok(chunk)) => {
                            buf.push_str(&String::from_utf8_lossy(&chunk));
                            queue.extend(parse_sse_events(&mut buf));
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

/// Consume every COMPLETE server-sent event in `buf` (events are separated by
/// a blank line), returning what they parse to and leaving any trailing partial
/// event in the buffer for the next network chunk. Pure, the Anthropic wire
/// handling is unit-testable without a socket.
pub fn parse_sse_events(buf: &mut String) -> Vec<StreamEvent> {
    let mut out = Vec::new();
    // Normalize CRLF once so the split below is simple.
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
            if data.is_empty() || data == "[DONE]" {
                continue;
            }
            let Ok(v) = serde_json::from_str::<serde_json::Value>(data) else {
                continue;
            };
            match v.get("type").and_then(|t| t.as_str()) {
                Some("content_block_delta") => {
                    if let Some(text) =
                        v.pointer("/delta/text")
                            .and_then(|t| t.as_str())
                            .filter(|_| {
                                v.pointer("/delta/type").and_then(|t| t.as_str())
                                    == Some("text_delta")
                            })
                    {
                        out.push(StreamEvent::TextDelta(text.to_string()));
                    }
                }
                Some("message_start") => {
                    let g = |p: &str| v.pointer(p).and_then(|n| n.as_u64()).unwrap_or(0);
                    out.push(StreamEvent::Usage(TokenUsage {
                        input_tokens: g("/message/usage/input_tokens"),
                        output_tokens: g("/message/usage/output_tokens"),
                        cache_creation_tokens: g("/message/usage/cache_creation_input_tokens"),
                        cache_read_tokens: g("/message/usage/cache_read_input_tokens"),
                    }));
                }
                Some("message_delta") => {
                    // Anthropic reports the terminal `stop_reason` here (not on
                    // `message_stop`); surface it so callers can distinguish a
                    // clean finish from a `max_tokens` cutoff.
                    if let Some(reason) = v.pointer("/delta/stop_reason").and_then(|r| r.as_str()) {
                        out.push(StreamEvent::StopReason(reason.to_string()));
                    }
                    let g = |p: &str| v.pointer(p).and_then(|n| n.as_u64()).unwrap_or(0);
                    out.push(StreamEvent::Usage(TokenUsage {
                        input_tokens: g("/usage/input_tokens"),
                        output_tokens: g("/usage/output_tokens"),
                        cache_creation_tokens: 0,
                        cache_read_tokens: 0,
                    }));
                }
                Some("message_stop") => out.push(StreamEvent::Done),
                Some("error") => {
                    let msg = v
                        .pointer("/error/message")
                        .and_then(|m| m.as_str())
                        .unwrap_or("upstream error");
                    out.push(StreamEvent::Error(msg.to_string()));
                }
                _ => {} // ping, content_block_start/stop, nothing to do
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sse_parser_handles_split_chunks_and_usage() {
        let mut buf = String::new();
        // First network chunk ends mid-event.
        buf.push_str(
            "event: message_start\n\
             data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":120,\"cache_read_input_tokens\":40}}}\n\n\
             event: content_block_delta\n\
             data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"The party \"}}\n\n\
             data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_del",
        );
        let events = parse_sse_events(&mut buf);
        assert_eq!(
            events,
            vec![
                StreamEvent::Usage(TokenUsage {
                    input_tokens: 120,
                    cache_read_tokens: 40,
                    ..Default::default()
                }),
                StreamEvent::TextDelta("The party ".into()),
            ],
        );
        assert!(buf.starts_with("data:"), "partial event stays buffered");

        // Second chunk completes it and finishes the stream.
        buf.push_str(
            "ta\",\"text\":\"fled.\"}}\n\n\
             data: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":77}}\n\n\
             data: {\"type\":\"message_stop\"}\n\n",
        );
        let events = parse_sse_events(&mut buf);
        assert_eq!(
            events,
            vec![
                StreamEvent::TextDelta("fled.".into()),
                StreamEvent::Usage(TokenUsage {
                    output_tokens: 77,
                    ..Default::default()
                }),
                StreamEvent::Done,
            ],
        );
        assert!(buf.is_empty());

        // Errors surface.
        let mut buf =
            "data: {\"type\":\"error\",\"error\":{\"message\":\"overloaded\"}}\n\n".to_string();
        assert_eq!(
            parse_sse_events(&mut buf),
            vec![StreamEvent::Error("overloaded".into())],
        );
    }

    #[test]
    fn message_delta_surfaces_max_tokens_stop_reason() {
        // A truncated answer: `stop_reason` rides the `message_delta` event and
        // must surface as StopReason ahead of the trailing usage.
        let mut buf = "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"max_tokens\"},\"usage\":{\"output_tokens\":2048}}\n\n".to_string();
        assert_eq!(
            parse_sse_events(&mut buf),
            vec![
                StreamEvent::StopReason("max_tokens".into()),
                StreamEvent::Usage(TokenUsage {
                    output_tokens: 2048,
                    ..Default::default()
                }),
            ],
        );
    }

    fn body(client: Client) -> serde_json::Value {
        client.request_body(
            "sys".into(),
            vec![serde_json::json!({"role": "user", "content": "hi"})],
        )
    }

    fn client(model: &str) -> Client {
        // reqwest is built without a crypto provider and panics on client
        // construction until one is installed; Err means one already is.
        let _ = rustls::crypto::ring::default_provider().install_default();
        Client::new("k", model).with_max_output_tokens(8192)
    }

    #[test]
    fn a_plain_request_sends_no_thinking_or_effort() {
        let b = body(client("claude-opus-5-5"));
        assert_eq!(b["max_tokens"], 8192);
        assert!(b.get("thinking").is_none());
        assert!(b.get("output_config").is_none());
        assert_eq!(b["stream"], true);
    }

    #[test]
    fn a_budget_never_reaches_a_model_that_rejects_it() {
        // Regression: `budget_tokens` 400s on every one of these, which
        // failed every request when a thinking budget was configured.
        for model in [
            "claude-opus-4-7",
            "claude-opus-4-8",
            "claude-opus-5",
            "claude-opus-5-5",
            "claude-sonnet-5",
            "claude-fable-5",
            "claude-fable-5-1",
        ] {
            let b = body(client(model).with_thinking_budget(4096));
            assert_eq!(
                b["thinking"],
                serde_json::json!({"type": "adaptive"}),
                "{model}"
            );
            assert!(!b.to_string().contains("budget_tokens"), "{model}");
            // The budget still buys headroom: adaptive thinking counts
            // toward max_tokens.
            assert_eq!(b["max_tokens"], 8192 + 4096, "{model}");
        }
    }

    #[test]
    fn the_4_6_models_get_adaptive_not_the_deprecated_budget() {
        for model in ["claude-opus-4-6", "claude-sonnet-4-6"] {
            let b = body(client(model).with_thinking_budget(2048));
            assert_eq!(
                b["thinking"],
                serde_json::json!({"type": "adaptive"}),
                "{model}"
            );
        }
    }

    #[test]
    fn budget_models_get_the_budget_form() {
        for model in ["claude-haiku-4-5", "claude-sonnet-4-5", "deepseek-v4-flash"] {
            let b = body(client(model).with_thinking_budget(2048));
            assert_eq!(
                b["thinking"],
                serde_json::json!({"type": "enabled", "budget_tokens": 2048}),
                "{model}"
            );
            // budget_tokens must stay below max_tokens; the answer keeps its cap.
            assert_eq!(b["max_tokens"], 8192 + 2048, "{model}");
        }
    }

    #[test]
    fn adaptive_intent_follows_the_model() {
        let b = body(client("claude-sonnet-5").with_adaptive_thinking());
        assert_eq!(b["thinking"], serde_json::json!({"type": "adaptive"}));
        // No budget given: the cap is untouched and covers the thinking.
        assert_eq!(b["max_tokens"], 8192);

        // Haiku 4.5 can't think adaptively, and there's no budget to fall
        // back on, so nothing is sent rather than a shape it rejects.
        let b = body(client("claude-haiku-4-5").with_adaptive_thinking());
        assert!(b.get("thinking").is_none());
        assert_eq!(b["max_tokens"], 8192);

        // Given a budget too, Haiku gets the budget form.
        let b = body(
            client("claude-haiku-4-5")
                .with_adaptive_thinking()
                .with_thinking_budget(1024),
        );
        assert_eq!(b["thinking"]["type"], "enabled");
        assert_eq!(b["thinking"]["budget_tokens"], 1024);
    }

    #[test]
    fn effort_rides_output_config_where_the_model_takes_it() {
        let b = body(client("claude-opus-5-5").with_effort(Effort::Low));
        assert_eq!(b["output_config"], serde_json::json!({"effort": "low"}));
        assert!(b.get("effort").is_none(), "effort is not top-level");

        // Clamped on 4.6, which has no xhigh.
        let b = body(client("claude-opus-4-6").with_effort(Effort::XHigh));
        assert_eq!(b["output_config"]["effort"], "high");

        // Errors on Haiku 4.5, so it's left off.
        let b = body(client("claude-haiku-4-5").with_effort(Effort::Max));
        assert!(b.get("output_config").is_none());
    }

    #[test]
    fn haiku_pricing_is_the_published_shape() {
        assert_eq!(HAIKU_4_5.input_cents_per_mtok, 100.0);
        assert_eq!(HAIKU_4_5.output_cents_per_mtok, 500.0);
    }
}
