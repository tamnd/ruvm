# vendor-qemu

These files are QEMU's, copied unmodified from the release named in `UPSTREAM`. ruvm reads them as data at build time and in tests: the QAPI schema, the decodetree files, every trace-events file, the three `.hx` tables, the expected ACPI tables from QEMU's test suite and the target configurations. They are under QEMU's licenses, which for these files is GPL-2.0-or-later unless a file says otherwise.

Do not edit anything here by hand. `cargo xtask upstream-sync <tag>` rewrites the directory from a QEMU checkout and prints what changed, and `cargo xtask vendor-check` fails CI if a file no longer matches `MANIFEST`. spec/24-workspace-layout.md says why the files are copied rather than submoduled, and spec/02-compat-contract.md says what happens when the tag moves.
