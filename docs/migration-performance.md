# Migration performance against QEMU

This note records how long a live migration takes and how long the guest is paused when ruvm is on one or both ends of the link, measured next to QEMU 11.1.0 on the same host. It is the third exit criterion of M5 (migration interop).

## Method

Both emulators were release builds: QEMU 11.1.0 built from the release tarball, and ruvm 0.4.5 built with `cargo build --release`. Source and destination ran on the same machine and talked over TCP on 127.0.0.1, so every direction used the same link. The four directions were QEMU to QEMU, QEMU to ruvm, ruvm to QEMU and ruvm to ruvm.

Each run started both sides with `-M q35 -nodefaults -accel tcg -display none`, a QMP socket and an `isa-debugcon` on port 0xe9. The source got `migrate-set-parameters` with `max-bandwidth` at 10 GiB/s, so the rate limit never applied, and `downtime-limit` at 300 ms. A script then issued `migrate`, polled `query-migrate` until it reported `completed`, and waited for the destination to leave `inmigrate`.

The guest is a 64 KiB `-bios` image written for this test rather than a Linux guest, so that the dirty rate is known and nothing else in the guest changes memory. It enters flat protected mode with interrupts off, fills a block of memory at 1 MiB with data that is not zero, prints a ready mark on the debug port, and then adds one to a dword in each page of a working set, over and over, printing a dot every 65536 page writes. The dots give the write rate before the migration and show that the guest keeps running on the destination afterwards.

There were two sizes and three dirty rates:

| Machine size | Data written at boot | Working set dirtied in a loop |
|---|---|---|
| 256 MiB | 32 MiB | none, 4 MiB, 32 MiB |
| 1 GiB | 256 MiB | none, 4 MiB, 32 MiB |

Every combination of size, working set and direction ran three times, 72 runs in all. The tables give the median with the lowest and highest of the three in brackets.

Three numbers are reported for each run. Total time is `total-time` from `query-migrate` on the source. Downtime is `downtime` from the same reply, which is what the source measures from stopping the guest to sending the last byte. Pause is the time from the `STOP` event on the source to the `RESUME` event on the destination, taken from the event timestamps. Pause is the one a guest user feels, because it includes the time the destination spends loading the last of the stream and its device state.

## The host, and how far to trust the numbers

There was no idle machine to run on. The runs were made on a shared server with 8 cores and 23 GiB of memory whose load average stayed between 65 and 105 for the whole session, from other builds and test runs. Neither side got a whole core for long, and the effective transfer rate on the loopback link was 10 to 50 MB/s rather than the gigabytes per second the link can do. That is why one run in the same cell can take three seconds and the next forty.

So the numbers below say which way things go and roughly by how much, but differences of less than a factor of two inside one cell are noise. They should be measured again on an idle host before anyone quotes them as figures. No run used KVM, because the host has no `/dev/kvm`.

## Results

Every one of the 72 migrations completed and every destination reached `running`. In 35 of the 36 runs with a working set, the destination guest went on printing dots after the switch. The exception was one ruvm to ruvm run at 256 MiB with the 32 MiB working set, whose destination printed nothing in the two seconds the script watched it; the other two runs of that cell kept going.

### 256 MiB machine

| Working set | Direction | Total time (s) | Downtime (ms) | Pause (ms) | MB sent |
|---|---|---|---|---|---|
| none | QEMU to QEMU | 4.7 (3.2, 40.7) | 20 (2, 90) | 80 (73, 962) | 34 |
| none | QEMU to ruvm | 2.4 (1.4, 37.0) | 14 (14, 41) | 288 (55, 4721) | 34 |
| none | ruvm to QEMU | 14.1 (5.7, 38.6) | 57 (5, 96) | 4517 (177, 15635) | 34 |
| none | ruvm to ruvm | 8.3 (4.3, 10.1) | 3 (2, 221) | 672 (449, 5591) | 34 |
| 4 MiB | QEMU to QEMU | 15.6 (2.6, 93.6) | 135 (74, 12914) | 311 (177, 12938) | 47 |
| 4 MiB | QEMU to ruvm | 4.1 (0.5, 75.8) | 161 (86, 779) | 2174 (1457, 5885) | 43 |
| 4 MiB | ruvm to QEMU | 5.0 (3.6, 93.4) | 333 (101, 516) | 452 (106, 491) | 52 |
| 4 MiB | ruvm to ruvm | 11.7 (6.3, 19.2) | 92 (59, 3935) | 596 (57, 4374) | 47 |
| 32 MiB | QEMU to QEMU | 10.0 (5.6, 58.8) | 623 (183, 879) | 825 (186, 890) | 111 |
| 32 MiB | QEMU to ruvm | 11.6 (9.6, 29.5) | 201 (52, 600) | 244 (53, 609) | 218 |
| 32 MiB | ruvm to QEMU | 20.4 (10.0, 237) | 1270 (730, 7780) | 1276 (750, 7818) | 198 |
| 32 MiB | ruvm to ruvm | 24.9 (18.3, 43.8) | 585 (440, 3423) | 813 (136, 3763) | 505 |

### 1 GiB machine

| Working set | Direction | Total time (s) | Downtime (ms) | Pause (ms) | MB sent |
|---|---|---|---|---|---|
| none | QEMU to QEMU | 36.0 (25.0, 39.4) | 33 (7, 274) | 292 (13, 988) | 271 |
| none | QEMU to ruvm | 16.8 (8.1, 17.9) | 6 (3, 15) | 19372 (12093, 44686) | 271 |
| none | ruvm to QEMU | 62.3 (43.8, 74.8) | 75 (3, 166) | 89 (5, 30771) | 271 |
| none | ruvm to ruvm | 40.8 (35.6, 47.0) | 45 (3, 93) | 5306 (27, 11444) | 271 |
| 4 MiB | QEMU to QEMU | 62.1 (40.3, 99.6) | 357 (96, 474) | 364 (201, 488) | 283 |
| 4 MiB | QEMU to ruvm | 34.9 (21.7, 43.7) | 12279 (104, 17802) | 12343 (149, 18161) | 282 |
| 4 MiB | ruvm to QEMU | 66.3 (44.8, 89.9) | 143 (107, 320) | 303 (121, 308) | 279 |
| 4 MiB | ruvm to ruvm | 43.7 (20.2, 55.6) | 148 (106, 380) | 1092 (266, 4774) | 280 |
| 32 MiB | QEMU to QEMU | 72.8 (44.9, 96.3) | 595 (192, 667) | 577 (195, 823) | 425 |
| 32 MiB | QEMU to ruvm | 50.5 (21.3, 59.8) | 703 (606, 1223) | 985 (807, 1169) | 363 |
| 32 MiB | ruvm to QEMU | 61.8 (46.5, 88.9) | 913 (329, 1126) | 966 (349, 1053) | 593 |
| 32 MiB | ruvm to ruvm | 45.1 (27.8, 55.3) | 499 (286, 737) | 613 (287, 993) | 620 |

### Guest write rate before the migration

The rate at which the guest dirtied pages before the migration started, in thousands of page writes per second, median of three:

| Working set | Machine | QEMU source | ruvm source |
|---|---|---|---|
| 4 MiB | 256 MiB | 108 to 196 | 3993 to 4103 |
| 4 MiB | 1 GiB | 109 to 152 | 8467 to 11945 |
| 32 MiB | 256 MiB | 303 to 415 | 65 to 152 |
| 32 MiB | 1 GiB | 306 to 371 | 153 to 238 |

The two numbers in each cell are the medians for the two destinations, since the rate was measured on the source before the destination started.

## What the numbers say

Total migration time is in the same range for all four directions. With every cell this noisy there is no direction that is reliably slower than QEMU to QEMU. A QEMU source sending to a ruvm destination was often the quickest of the four, and the same amount of data crossed the link in each direction for the idle and 4 MiB cases. The 32 MiB working set sent more data when ruvm was the source because ruvm needed more dirty sync rounds to converge, up to 68 in one run against at most 24 for QEMU.

Downtime as the source reports it is within the 300 ms limit or close to it in most cells, as it is for QEMU. The overshoots, some of them seconds long, come from runs where the source thread lost the CPU in the last round, and they happen on both emulators.

Pause is where ruvm differs. When ruvm is the destination, the guest restarts later than QEMU's would, by up to tens of seconds on the 1 GiB idle machine, even though the source has already reported completion with a downtime of a few milliseconds. The source has sent everything by then, so the time goes on the ruvm destination working through the end of the stream before it resumes the guest. On this starved host that gap is the largest difference between ruvm and QEMU.

A profile of the ruvm destination during that gap, on the 1 GiB idle machine with a QEMU source, shows where the time goes. The incoming migration thread is runnable the whole time and spends nearly all of it in the RAM section loader and the page faults it causes. For each zero page, the loader copies the page out of guest RAM one byte at a time, because `RamBlock::read` loads each byte as an atomic, and then checks the copy for a byte that is not zero, again one byte at a time. QEMU checks the page in place with a vectorised `buffer_is_zero()`. The zero page records are only a few bytes each, so they reach the destination long before it has processed them, and the source reports completion while the destination still has hundreds of megabytes of zero pages to check. Reading and checking whole words, or checking the page in place as QEMU does, should close most of the gap. That change is not part of this measurement.

A ruvm source running the guest under TCG ran the 4 MiB loop 20 to 100 times faster than QEMU did, so ruvm had to migrate a much harder guest in those cells and still converged in a similar time. With the 32 MiB working set the ordering flipped and the ruvm guest wrote more slowly than QEMU's.

## Reproducing

The guest is assembled with `as --32 --defsym FILL_PAGES=N --defsym WSS_PAGES=M`, linked as a flat 64 KiB binary at 0xFFFF0000, where the firmware image sits below 4 GiB, and passed with `-bios`. The driver is a short Python script that starts both sides, sets the parameters above over QMP, runs the migration and collects `query-migrate`, the two events and the debug port output. Run it on a host with nothing else on it, with source and destination pinned to separate cores, and with `-accel kvm` as well as TCG once a KVM host is available.
