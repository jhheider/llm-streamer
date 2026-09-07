//! Stream one short completion against a real provider, and print every event.
//!
//! The unit tests pin the wire grammar against fixtures; this hits the actual
//! gateway, which is the only way to catch a provider changing its shape.
//!
//! ```sh
//! AI_PROVIDER=openrouter \
//! AI_API_KEY=sk-or-… \
//! AI_MODEL=deepseek/deepseek-v4-flash-0731 \
//!   cargo run --example smoke
//! ```
//!
//! Watch for a `Cost` line: that is the gateway reporting what it billed, which
//! is the figure to meter against when a provider sends one.

use futures::StreamExt;
use llm_streamer::{Client, StreamEvent, Wire, anthropic, openai};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // reqwest is built with `rustls-no-provider`, so someone has to install a
    // CryptoProvider before the first Client is constructed. Err just means one
    // is already installed.
    let _ = rustls::crypto::ring::default_provider().install_default();

    let provider = std::env::var("AI_PROVIDER").unwrap_or_else(|_| "openrouter".into());
    let api_key = std::env::var("AI_API_KEY").map_err(|_| "set AI_API_KEY")?;
    let model = std::env::var("AI_MODEL").map_err(|_| "set AI_MODEL")?;

    let spec = llm_streamer::provider_spec(&provider)
        .ok_or_else(|| format!("unknown AI_PROVIDER {provider:?}"))?;
    let base_url = std::env::var("AI_BASE_URL").unwrap_or_else(|_| spec.base_url.to_string());

    println!(
        "provider={} wire={:?} base={base_url} model={model}",
        spec.name, spec.wire
    );

    let client: Client = match spec.wire {
        Wire::Anthropic => anthropic::Client::new(api_key, &model)
            .with_base_url(base_url)
            .with_max_output_tokens(64)
            .into(),
        Wire::OpenAi => openai::Client::new(base_url, api_key, &model)
            .with_max_output_tokens(64)
            .into(),
    };

    let stream = client
        .stream_chat(
            "You are terse.".into(),
            vec![serde_json::json!({"role": "user", "content": "Name three colours."})],
        )
        .await?;
    futures::pin_mut!(stream);

    let mut usage = llm_streamer::TokenUsage::default();
    let mut reported: Option<f64> = None;
    while let Some(ev) = stream.next().await {
        match ev {
            StreamEvent::TextDelta(t) => print!("{t}"),
            StreamEvent::Usage(u) => {
                usage = llm_streamer::merge_usage(usage, u);
                println!("\n[usage] {u:?}");
            }
            StreamEvent::Cost(c) => {
                reported = Some(c);
                println!("[cost] {c:.6}c reported by the gateway");
            }
            StreamEvent::StopReason(r) => println!("\n[stop] {r}"),
            StreamEvent::Done => {
                println!("\n[done]");
                break;
            }
            StreamEvent::Error(e) => {
                eprintln!("\n[error] {e}");
                std::process::exit(1);
            }
        }
    }

    match reported {
        Some(c) => println!(
            "charged {}c (gateway-reported)",
            llm_streamer::reported_cost_cents(c)
        ),
        None => println!("no cost reported; price it from a configured `Pricing`"),
    }
    println!("final usage: {usage:?}");
    Ok(())
}
