# llm-streamer

Streaming LLM chat with **one event shape across providers**, plus built-in
token-cost accounting. Unofficial.

- **One shape to match on.** `stream_chat` yields `StreamEvent`s — `TextDelta`,
  `Usage`, `StopReason`, `Cost`, `Done`, `Error` — whatever backend produced
  them. Wire faults become `Error`, so you handle exactly one thing.
- **Two wire formats, many providers.** `anthropic` speaks the Anthropic
  Messages API (`x-api-key`); `openai` speaks chat-completions with bearer auth,
  which covers OpenAI, OpenRouter, DeepSeek, Groq, Together and xAI. Ollama's
  OpenAI-compatible endpoint works unchanged, so the same code runs locally.
  `provider_spec` resolves a preset name to a base URL and wire.
- **Cost accounting built in.** `Pricing` + `usage_cost_cents` turn reported
  usage into whole cents, conservatively (rounded up, floor of 1) — for products
  that meter AI spend. When a gateway bills in-band, as OpenRouter does, the
  stream hands you a `Cost` event instead; prefer it, since a configured price
  drifts silently every time a provider changes its rate card.
- **Pure, testable wire parsing.** Each backend's SSE handling is a plain
  function over a buffer, so split-chunk behaviour is unit-tested without a
  socket. The OpenRouter fixtures are captured from the live wire, which is how
  they cover the things a hand-written fixture misses — `finish_reason` arriving
  twice, `delta.reasoning` that must not reach the answer, and
  `completion_tokens` including reasoning tokens.

## Usage

```rust,no_run
use futures::StreamExt;
use llm_streamer::{StreamEvent, anthropic::Client};

# async fn ex() -> Result<(), String> {
let client = Client::new(std::env::var("ANTHROPIC_API_KEY").unwrap(), "claude-haiku-4-5");
let stream = client
    .stream_chat(
        "You are terse.".into(),
        vec![serde_json::json!({"role": "user", "content": "Hi"})],
    )
    .await?;
futures::pin_mut!(stream);
while let Some(ev) = stream.next().await {
    match ev {
        StreamEvent::TextDelta(t) => print!("{t}"),
        StreamEvent::Done => break,
        StreamEvent::Error(e) => return Err(e),
        _ => {}
    }
}
# Ok(()) }
```

The `smoke` example streams one completion against a real gateway and prints
every event, which is the only way to catch a provider changing its shape:

```sh
AI_PROVIDER=openrouter AI_API_KEY=… AI_MODEL=… cargo run --example smoke
```

## TLS

This crate does not choose a crypto provider. `reqwest` is built with
`rustls-no-provider` to keep `aws-lc-rs` and its cmake C build out of your tree,
so install one before the first request or the client panics on construction:

```rust,ignore
let _ = rustls::crypto::ring::default_provider().install_default();
```

## License

MIT OR Apache-2.0.
