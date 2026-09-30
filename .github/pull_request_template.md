## What this changes

<!-- The problem first, then the change. What is in the diff does not need restating. -->

## Why

<!-- If it closes an issue, say `Closes #N`. If it implements part of a milestone, say which. -->

## How it was verified

<!-- Which test fails without this change. If none does, say so and say why. -->

## Checklist

- [ ] `cargo xtask ci` passes locally
- [ ] A change a guest, a management tool or a migration stream can see names the QEMU 11.1 source it matches
- [ ] A deliberate difference from QEMU has an entry in `conformance/divergences.toml`
- [ ] A new device or VMState section comes with a migration round trip against QEMU 11.1
- [ ] A new `unsafe` block has a `SAFETY` comment and fits the crate's budget, or the budget change is called out
- [ ] Code ported from QEMU lands in a GPL-2.0-or-later crate
- [ ] A performance claim comes with the command that reproduces it and the machine it ran on
- [ ] Prose follows the house rules: plain English, no em dashes, no horizontal rules, no hard-wrapped sentences
