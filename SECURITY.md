# Security

ruvm runs untrusted guest code and parses untrusted guest data, so a bug in a device model can be a guest escape. Please report anything like that privately.

## How to report

Use [GitHub's private vulnerability reporting](https://github.com/tamnd/ruvm/security/advisories/new). Include the command line, the guest if you can share it, and what the guest can do that it should not. You will get an answer within a week.

## What counts

The threat model is in [`spec/19-security-and-confidential.md`](spec/19-security-and-confidential.md). In short: with KVM, HVF or WHPX, anything a guest can do to crash ruvm, read or write host memory outside its RAM, or run code in the ruvm process is in scope. Under TCG, the guest is treated as trusted for the purposes of a security report, which is the same position QEMU takes, though we still want to hear about it as an ordinary bug.

A bug that is also present in QEMU 11.1 should be reported to QEMU as well. We will coordinate the disclosure with them.

## Supported versions

Before 1.0 only the latest release gets fixes.
