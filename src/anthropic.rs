//! The Anthropic Messages API backend (`/v1/messages`).
//!
//! Endpoint-agnostic within the Anthropic wire format: point [`Client`] at
//! Anthropic directly, or at any compatible gateway, DeepSeek documents
//! `base_url = https://api.deepseek.com/anthropic`, and Grok exposes a similar
//! compat surface. Yields the crate's neutral [`StreamEvent`](crate::StreamEvent)s.

use crate::{StreamEvent, TokenUsage};
use futures::StreamExt;
use std::collections::VecDeque;

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
    thinking_budget_tokens: Option<u32>,
    http: reqwest::Client,
}

impl Client {
    /// A client for `model`, talking to Anthropic directly. Defaults:
    /// `base_url = https://api.anthropic.com`, `max_output_tokens = 4096`,
    /// no extended thinking.
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Client {
            base_url: "https://api.anthropic.com".into(),
            api_key: api_key.into(),
            model: model.into(),
            max_output_tokens: 4096,
            thinking_budget_tokens: None,
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
    /// the system prompt shapes real length.
    pub fn with_max_output_tokens(mut self, n: u32) -> Self {
        self.max_output_tokens = n;
        self
    }

    /// Request extended thinking with `budget` tokens (Anthropic wire shape;
    /// DeepSeek's compat endpoint honors it). Thinking deltas are parsed and
    /// dropped, only answer text streams out. Thinking bills at the output
    /// rate and adds latency, so it's off by default.
    pub fn with_thinking_budget(mut self, budget: u32) -> Self {
        self.thinking_budget_tokens = Some(budget);
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

    /// One streamed chat completion. `messages` are `{"role", "content"}` JSON
    /// objects, most-recent last; `system` is the system prompt. Yields parsed
    /// [`StreamEvent`]s; wire faults surface as [`StreamEvent::Error`] so
    /// callers handle exactly one shape.
    pub async fn stream_chat(
        &self,
        system: String,
        messages: Vec<serde_json::Value>,
    ) -> Result<impl futures::Stream<Item = StreamEvent> + Send + 'static, String> {
        let mut body = serde_json::json!({
            "model": self.model,
            "max_tokens": self.max_output_tokens,
            "system": system,
            "messages": messages,
            "stream": true,
        });
        if let Some(budget) = self.thinking_budget_tokens {
            // Anthropic semantics: budget_tokens must stay below max_tokens,
            // so thinking gets its own headroom on top of the answer's.
            body["max_tokens"] = (self.max_output_tokens + budget).into();
            body["thinking"] = serde_json::json!({
                "type": "enabled",
                "budget_tokens": budget,
            });
        }
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

    #[test]
    fn haiku_pricing_is_the_published_shape() {
        assert_eq!(HAIKU_4_5.input_cents_per_mtok, 100.0);
        assert_eq!(HAIKU_4_5.output_cents_per_mtok, 500.0);
    }
}
