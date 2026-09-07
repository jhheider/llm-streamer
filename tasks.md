# llm-streamer tasks

## test

Run the test suite. The unit tests pin the SSE grammar against fixtures and
need no network.

```bash
cargo test --all-features
```

## lint

fmt + clippy the way CI does: warnings are errors.

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
```

## check

The full local gate: lint, then test, then a real build.

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
cargo build --all-features
```

## smoke

Stream one completion against a real gateway and print every event. Needs
AI_API_KEY and AI_MODEL in the environment; AI_PROVIDER defaults to openrouter.

```bash
cargo run --example smoke
```

## msrv

Verify the declared rust-version still builds. Slow: it bisects real toolchains.

```bash
cargo msrv verify
```
