# Changelog

Notable changes, newest first. This project is pre-1.0 and makes no compatibility promise about its own APIs until it has one. The compatibility it does promise is with QEMU 11.1, and each release says how much of that is real.

The minor version is the number of milestones finished. 0.1.0 is the release where M0 closes, 0.2.0 where M1 closes, and so on up to M12, which is 1.0. Patch releases come whenever enough has landed to be worth a tag. The milestones are the issues labeled `kind/milestone` at https://github.com/tamnd/ruvm/issues.

## Unreleased

## 0.1.0

M0 is done. There is no emulator yet, but everything the emulator will be built inside is in place and checked on every pull request.

The workspace has all 107 crates from the catalog in `spec/24-workspace-layout.md`, each with its license, its layer and an unsafe budget of zero. `cargo xtask ci` runs the layer rule, the license provenance rule, the unsafe audit, the prose rules, the vendored input check, rustfmt, clippy, the tests and the docs, and CI runs the same thing on Linux x86_64 and aarch64, macOS and Windows, along with an MSRV build on 1.85, cargo-deny and actionlint with zizmor for the workflows (#14, #15).

`ruvm` dispatches on argv[0] to all 29 system emulators, the 38 user mode emulators and the 11 tools, and prints the same version text QEMU 11.1.0 prints for each one, so libvirt's version probe parses it (#16). Everything past `--version` exits with an error that says it is not implemented yet.

`vendor-qemu/` holds the QAPI schemas, the decodetree files, the trace-events files, the hx files, the ACPI expected tables and the target list from QEMU v11.1.0, with a manifest of their hashes. `cargo xtask upstream-sync <tag>` refreshes them and prints what changed (#17).

Tags now produce a release with attested archives for Linux, macOS and Windows, and `cargo xtask version` sets the version everywhere at once (#20).

The specification, the README and the license files came before all of this and are unchanged.
