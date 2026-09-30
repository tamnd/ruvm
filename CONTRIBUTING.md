# Contributing

Thank you for looking. This file is short on purpose. The design is in [`spec/`](spec/), and when the code and the spec disagree the code is wrong until a pull request changes the spec.

## Before you start

Pick an item from a milestone issue, or open an issue first if the change is large. A change that adds a device, a machine type, a QMP command or a command line option should name the QEMU 11.1 source file it tracks, because compatibility is the first thing it will be reviewed for.

## The rules a pull request is checked against

`cargo xtask ci` runs everything CI runs: formatting, clippy with warnings denied, the layer rule, the license provenance rule, the unsafe budgets, the prose rules and the tests. Run it before you push.

The layer rule is in [`spec/24-workspace-layout.md`](spec/24-workspace-layout.md). A crate may depend on crates in its own layer or below, never above. A new crate needs a line in `xtask/layers.toml`.

Every `unsafe` block carries a `// SAFETY:` comment that says why it is sound. A crate may only hold as many unsafe blocks as its budget in `Cargo.toml` allows, and raising a budget is a reviewed change.

A permissive crate (MIT OR Apache-2.0) must not contain code ported from QEMU. If you port logic from QEMU, the crate it lands in must be GPL-2.0-or-later.

Behavior that a guest, a management tool or a migration stream can observe must match QEMU 11.1 unless it is listed in `conformance/divergences.toml` with a reason from [`spec/02-compat-contract.md`](spec/02-compat-contract.md) section 2.

## Prose

Documentation, pull request descriptions, issue comments and commit messages follow the same rules:

- Plain English, written the way you would explain it to a colleague.
- No em dashes and no en dashes. Use a full stop or a comma.
- No horizontal rules.
- One paragraph is one line. Do not wrap sentences by hand.

`cargo xtask style` checks the markdown files in the repository.

## Commits and pull requests

One logical change per pull request. The title says what changes in the imperative, like "Add the PL011 UART". The body says what was wrong or missing, what the change does, and how it was tested. If it implements part of a milestone, say which one, and the milestone issue gets its checkbox ticked when the pull request merges.

## Releases

`cargo xtask version 0.1.1` moves the workspace version and every internal pin in one go. Add a `## 0.1.1` section to the changelog in the same pull request, merge it, then push the tag `v0.1.1`. The release workflow refuses a tag that does not match the version in `Cargo.toml` or that has no changelog section, builds the archives for Linux, macOS and Windows, attests them and publishes the release with that changelog section as its notes.
