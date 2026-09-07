//! Streaming LLM chat, one event shape across providers.
//!
//! The **provider-neutral core** lives here: [`StreamEvent`] (what a streamed
//! completion emits, text, usage, done, error), [`TokenUsage`], and cost
//! accounting ([`Pricing`] + [`usage_cost_cents`]) that turns reported usage
//! into whole cents, conservatively, for products that meter AI spend.
//!
//! Each **provider backend** is a module that speaks one wire format and yields
//! these same neutral events: [`anthropic`] (Messages API, `x-api-key`) and
//! [`openai`] (chat completions, bearer auth), the latter covering OpenAI,
//! OpenRouter, DeepSeek, Groq, Together and xAI. [`Client`] dispatches over
//! both so callers pick a provider at runtime; [`provider_spec`] resolves a
//! preset name to a base URL and [`Wire`].
//!
//! Cost is reported two ways. Compute it from configured [`Pricing`] with
//! [`usage_cost_cents`], or take the [`StreamEvent::Cost`] the stream hands
//! you when a gateway bills in-band, as OpenRouter does. Prefer the reported
//! figure: it cannot drift out of date.
//!
//! # TLS
//!
//! This crate does not choose a crypto provider. If your `reqwest` is built
//! with `rustls-no-provider` (the way to keep `aws-lc-rs` and its cmake C
//! build out of a tree), install one before the first request or the client
//! panics on construction:
//!
//! ```ignore
//! let _ = rustls::crypto::ring::default_provider().install_default();
//! ```
//!
//! ```no_run
//! # async fn ex() -> Result<(), String> {
//! use llm_streamer::{StreamEvent, anthropic::Client};
//! use futures::StreamExt;
//!
//! let client = Client::new(std::env::var("ANTHROPIC_API_KEY").unwrap(), "claude-haiku-4-5");
//! let stream = client
//!     .stream_chat(
//!         "You are terse.".into(),
//!         vec![serde_json::json!({"role": "user", "content": "Hi"})],
//!     )
//!     .await?;
//! futures::pin_mut!(stream);
//! while let Some(ev) = stream.next().await {
//!     match ev {
//!         StreamEvent::TextDelta(t) => print!("{t}"),
//!         StreamEvent::Done => break,
//!         StreamEvent::Error(e) => return Err(e),
//!         StreamEvent::Usage(_) | StreamEvent::StopReason(_) | StreamEvent::Cost(_) => {}
//!     }
//! }
//! # Ok(()) }
//! ```

use serde::{Deserialize, Serialize};

pub mod anthropic;
pub mod openai;

// ─── Stream events (provider-neutral) ───────────────────────────────────────

/// One parsed event off a streamed completion, the same shape whatever
/// backend produced it.
#[derive(Clone, Debug, PartialEq)]
pub enum StreamEvent {
    /// A chunk of assistant text.
    TextDelta(String),
    /// Usage numbers. A backend may emit these more than once (e.g. input side
    /// then output side); [`merge_usage`] combines them.
    Usage(TokenUsage),
    /// Why the model stopped, as the backend reported it (Anthropic:
    /// `"end_turn"`, `"max_tokens"`, `"stop_sequence"`, …). Emitted before
    /// [`Done`](StreamEvent::Done) when the backend surfaces it; a caller can
    /// tell "the model finished" from "the model was cut off at the token cap"
    /// (`"max_tokens"`).
    StopReason(String),
    /// What the gateway says it actually billed for this request, in cents.
    ///
    /// Only some backends report it (OpenRouter does, on the terminal usage
    /// chunk). Prefer it over computing from [`Pricing`] when present: a
    /// reported figure is what you were really charged, while configured
    /// prices drift silently every time a provider changes its rate card.
    /// [`reported_cost_cents`] applies the same rounding policy as
    /// [`usage_cost_cents`].
    Cost(f64),
    /// The stream ended normally.
    Done,
    /// The upstream reported an error.
    Error(String),
}

// ─── Usage & cost (provider-neutral) ────────────────────────────────────────

/// Token counts from one model response.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_creation_tokens: u64,
    pub cache_read_tokens: u64,
}

/// Merge usage fragments by taking per-field maxima, never undercount when a
/// backend reports usage in pieces.
pub fn merge_usage(a: TokenUsage, b: TokenUsage) -> TokenUsage {
    TokenUsage {
        input_tokens: a.input_tokens.max(b.input_tokens),
        output_tokens: a.output_tokens.max(b.output_tokens),
        cache_creation_tokens: a.cache_creation_tokens.max(b.cache_creation_tokens),
        cache_read_tokens: a.cache_read_tokens.max(b.cache_read_tokens),
    }
}

/// Per-model token prices, in cents per million tokens. Prices aren't reported
/// in-band by any provider, so they're your configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pricing {
    pub input_cents_per_mtok: f64,
    pub output_cents_per_mtok: f64,
    pub cache_write_cents_per_mtok: Option<f64>,
    pub cache_read_cents_per_mtok: Option<f64>,
}

/// Whole cents, rounded up, min 1, conservative so usage never undercounts.
/// When cache prices are absent they fall back to input×1.25 (write) and
/// input×1 (read), the common public multipliers.
pub fn usage_cost_cents(usage: TokenUsage, pricing: &Pricing) -> i32 {
    let per_tok = |cents_per_mtok: f64| cents_per_mtok / 1_000_000.0;
    let input = per_tok(pricing.input_cents_per_mtok);
    let output = per_tok(pricing.output_cents_per_mtok);
    let cache_write = pricing
        .cache_write_cents_per_mtok
        .map(per_tok)
        .unwrap_or(input * 1.25);
    let cache_read = pricing
        .cache_read_cents_per_mtok
        .map(per_tok)
        .unwrap_or(input);
    let cost = usage.input_tokens as f64 * input
        + usage.cache_creation_tokens as f64 * cache_write
        + usage.cache_read_tokens as f64 * cache_read
        + usage.output_tokens as f64 * output;
    (cost.ceil() as i32).max(1)
}

/// Round a gateway-reported cost (cents, fractional) the same way
/// [`usage_cost_cents`] rounds a computed one: up, floor of 1, so a charge is
/// never recorded as free.
pub fn reported_cost_cents(cents: f64) -> i32 {
    (cents.ceil() as i32).max(1)
}

// ─── Provider presets ───────────────────────────────────────────────────────

/// Which wire format a provider speaks. The two are not interchangeable:
/// they differ in endpoint, auth header, request shape and SSE grammar.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Wire {
    /// Anthropic Messages (`/v1/messages`, `x-api-key`).
    Anthropic,
    /// OpenAI chat completions (`/chat/completions`, bearer).
    OpenAi,
}

/// A known gateway: the base URL and wire format to reach it.
///
/// This is deliberately NOT a model/price registry. Prices belong to
/// configuration (or come back in-band via [`StreamEvent::Cost`]) because a
/// table compiled into a binary goes stale the moment a provider re-prices,
/// and a stale price in a metered product is a billing bug.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProviderSpec {
    pub name: &'static str,
    pub base_url: &'static str,
    pub wire: Wire,
}

/// Every preset, in the order they are offered to a user.
pub const PROVIDERS: &[ProviderSpec] = &[
    ProviderSpec {
        name: "anthropic",
        base_url: "https://api.anthropic.com",
        wire: Wire::Anthropic,
    },
    ProviderSpec {
        name: "openrouter",
        base_url: "https://openrouter.ai/api/v1",
        wire: Wire::OpenAi,
    },
    ProviderSpec {
        name: "openai",
        base_url: "https://api.openai.com/v1",
        wire: Wire::OpenAi,
    },
    // DeepSeek fronts both wires. The OpenAI-flavor one is its documented
    // default; `deepseek-anthropic` is the /anthropic compat surface.
    ProviderSpec {
        name: "deepseek",
        base_url: "https://api.deepseek.com/v1",
        wire: Wire::OpenAi,
    },
    ProviderSpec {
        name: "deepseek-anthropic",
        base_url: "https://api.deepseek.com/anthropic",
        wire: Wire::Anthropic,
    },
    ProviderSpec {
        name: "xai",
        base_url: "https://api.x.ai/v1",
        wire: Wire::OpenAi,
    },
    ProviderSpec {
        name: "groq",
        base_url: "https://api.groq.com/openai/v1",
        wire: Wire::OpenAi,
    },
    ProviderSpec {
        name: "together",
        base_url: "https://api.together.xyz/v1",
        wire: Wire::OpenAi,
    },
];

/// Look a preset up by name, case-insensitively.
pub fn provider_spec(name: &str) -> Option<&'static ProviderSpec> {
    let name = name.trim().to_ascii_lowercase();
    PROVIDERS.iter().find(|p| p.name == name)
}

// ─── One client over either wire ────────────────────────────────────────────

/// A backend-agnostic client, so callers pick a provider at runtime without
/// branching at every call site.
#[derive(Clone)]
pub enum Client {
    Anthropic(anthropic::Client),
    OpenAi(openai::Client),
}

impl Client {
    /// The model this client targets.
    pub fn model(&self) -> &str {
        match self {
            Client::Anthropic(c) => c.model(),
            Client::OpenAi(c) => c.model(),
        }
    }

    /// Which wire format this client speaks.
    pub fn wire(&self) -> Wire {
        match self {
            Client::Anthropic(_) => Wire::Anthropic,
            Client::OpenAi(_) => Wire::OpenAi,
        }
    }

    /// One streamed chat completion, yielding the same [`StreamEvent`]s
    /// whichever backend is underneath. The stream is boxed because the two
    /// backends return different concrete types.
    pub async fn stream_chat(
        &self,
        system: String,
        messages: Vec<serde_json::Value>,
    ) -> Result<futures::stream::BoxStream<'static, StreamEvent>, String> {
        use futures::StreamExt;
        match self {
            Client::Anthropic(c) => Ok(c.stream_chat(system, messages).await?.boxed()),
            Client::OpenAi(c) => Ok(c.stream_chat(system, messages).await?.boxed()),
        }
    }
}

impl From<anthropic::Client> for Client {
    fn from(c: anthropic::Client) -> Self {
        Client::Anthropic(c)
    }
}

impl From<openai::Client> for Client {
    fn from(c: openai::Client) -> Self {
        Client::OpenAi(c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn u(i: u64, o: u64, cw: u64, cr: u64) -> TokenUsage {
        TokenUsage {
            input_tokens: i,
            output_tokens: o,
            cache_creation_tokens: cw,
            cache_read_tokens: cr,
        }
    }

    #[test]
    fn cost_math_is_conservative_with_cache_tiers() {
        // $1/MTok in, $5/MTok out, cache write 1.25× / read 0.1×.
        let p = Pricing {
            input_cents_per_mtok: 100.0,
            output_cents_per_mtok: 500.0,
            cache_write_cents_per_mtok: Some(125.0),
            cache_read_cents_per_mtok: Some(10.0),
        };
        assert_eq!(usage_cost_cents(u(1_000_000, 0, 0, 0), &p), 100);
        assert_eq!(usage_cost_cents(u(0, 1_000_000, 0, 0), &p), 500);
        assert_eq!(usage_cost_cents(u(0, 0, 1_000_000, 0), &p), 125);
        assert_eq!(usage_cost_cents(u(0, 0, 0, 1_000_000), &p), 10);
        assert_eq!(usage_cost_cents(u(10, 10, 0, 0), &p), 1); // floor of 1 cent
    }

    #[test]
    fn cost_math_falls_back_when_cache_prices_absent() {
        // DeepSeek-style: no cache prices → write 1.25× in, read 1× in.
        let p = Pricing {
            input_cents_per_mtok: 28.0,
            output_cents_per_mtok: 42.0,
            cache_write_cents_per_mtok: None,
            cache_read_cents_per_mtok: None,
        };
        assert_eq!(usage_cost_cents(u(1_000_000, 0, 0, 0), &p), 28);
        assert_eq!(usage_cost_cents(u(0, 1_000_000, 0, 0), &p), 42);
        assert_eq!(usage_cost_cents(u(0, 0, 1_000_000, 0), &p), 35); // 1.25×28
        assert_eq!(usage_cost_cents(u(0, 0, 0, 1_000_000), &p), 28); // 1× fallback
    }

    #[test]
    fn reported_cost_rounds_up_and_never_reads_as_free() {
        assert_eq!(reported_cost_cents(0.95), 1); // sub-cent charge still bills 1
        assert_eq!(reported_cost_cents(1.0), 1);
        assert_eq!(reported_cost_cents(1.01), 2);
        assert_eq!(reported_cost_cents(0.0), 1);
    }

    #[test]
    fn provider_presets_resolve_and_carry_the_right_wire() {
        let or = provider_spec("openrouter").expect("openrouter preset");
        assert_eq!(or.wire, Wire::OpenAi);
        assert_eq!(or.base_url, "https://openrouter.ai/api/v1");
        // Case-insensitive, and unknown names are None so callers can fall
        // back to an explicit base URL rather than silently guessing.
        assert!(provider_spec("OpenRouter").is_some());
        assert!(provider_spec("nope").is_none());
        // DeepSeek is reachable over either wire; the names must not collide.
        assert_eq!(provider_spec("deepseek").unwrap().wire, Wire::OpenAi);
        assert_eq!(
            provider_spec("deepseek-anthropic").unwrap().wire,
            Wire::Anthropic
        );
    }

    #[test]
    fn every_preset_is_uniquely_named() {
        let mut names: Vec<_> = PROVIDERS.iter().map(|p| p.name).collect();
        names.sort_unstable();
        let before = names.len();
        names.dedup();
        assert_eq!(names.len(), before, "preset names must be unique");
    }

    #[test]
    fn merge_takes_per_field_maxima() {
        let merged = merge_usage(u(120, 0, 0, 40), u(0, 77, 0, 0));
        assert_eq!(merged, u(120, 77, 0, 40));
    }
}
