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

## release-check

Run before tagging a release. Fails unless the current commit is on
`origin/main` and no tag for the version in `Cargo.toml` exists yet, so a tag
cannot land on a branch commit or be moved onto an existing one.

```bash
set -eu
v="v$(sed -n 's/^version *= *"\(.*\)"/\1/p' Cargo.toml | head -1)"
git fetch --tags --quiet origin
git branch -r --contains HEAD | grep -qx '  origin/main' || {
  echo "HEAD is not on origin/main; tag only a merged release commit"; exit 1; }
git rev-parse -q --verify "refs/tags/$v" >/dev/null && {
  echo "tag $v already exists; never move a tag, release a new version"; exit 1; }
echo "ok to tag $v on $(git rev-parse --short HEAD)"
```
