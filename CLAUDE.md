# CLAUDE.md

`llm-streamer` is a small streaming client for OpenAI-compatible and Anthropic
chat APIs. It parses server-sent events into a `StreamEvent` stream and keeps
tool-call accumulation next to the parsing that produces it. **This repo is
public**, which adds rules below to the usual ones.

This file holds the rules for every agent working here, whatever its name.
`AGENTS.md` points here; keep the rules in this one file.

## The five rules

Short form. The long form, the reasoning, and the missteps that produced them
are in [`jhheider/epsilon`'s `CLAUDE.md`](https://github.com/jhheider/epsilon/blob/main/CLAUDE.md).

1. **Never cheat to pass a test.** A test is the spec. The only edit allowed to
   an existing test is removing its `ignore` once it passes. If a test looks
   wrong, say why in the PR and get Jacob's agreement before touching it.
2. **No dead code, and no hiding it.** Nothing that is uncalled stays. No
   `allow(dead_code)`, no `_` prefixes to silence a lint, no `let _ =`.
3. **One way to do a thing.** Ask the same functions the runtime asks; do not
   write a second list, a second lookup, or a second copy of a default.
4. **Build and check it before you claim it.** Every claim in a PR body has to
   be backed by output pasted from a real run.
5. **One task per PR, from an up-to-date `main`.** Never branch from another
   task's unmerged branch.

## Because this repo is public

- **No `Claude-Session:` trailer** and **no session URL from an assistant
  console** in commits, PR bodies, PR comments or code. Keep
  `Co-Authored-By` trailers.
- **CI runs on hosted runners only** (ubuntu, macos, windows). Do not add a
  self-hosted runner to this repo.
- Nothing that identifies a private project's internals goes in a commit
  message, a test fixture or a doc comment.

## The gate

```sh
mdtask check
```

That runs `fmt`, `clippy -D warnings`, the unit tests, the examples and the
MSRV check. Run it before every push. A branch must not contain "cargo fmt" or
"clippy fix" commits when the PR opens; fix formatting in the commit that
caused it.

`main` is protected: every change is a PR, with green CI, a merge commit, and
no force pushes. Never work around the protection.

## Releasing

Tags are the only release mechanism; this crate is not published to crates.io,
so a wrong tag cannot be withdrawn. The procedure:

1. The release PR sets `version` in `Cargo.toml`. Nothing else may be in that
   PR except the changelog entry.
2. Wait for it to merge.
3. Tag **the merge commit that is already on `main`**, not the branch commit:

   ```sh
   git -C ~/code/llm-streamer fetch --tags origin
   git -C ~/code/llm-streamer checkout main
   git -C ~/code/llm-streamer pull
   cargo xtask release-check   # or the tasks.md recipe: fails unless HEAD is on origin/main and the tag is free
   git -C ~/code/llm-streamer tag -a v<version> -m "v<version>"
   git -C ~/code/llm-streamer push origin v<version>
   ```

4. **Never move an existing tag.** If a release is wrong, ship a new version.

Before tagging, confirm the tag will land on `main` and that the version in
`Cargo.toml` matches the tag you are about to cut. The `release-check` recipe
in `tasks.md` checks both.
