# 07. ruvm-jit: IR and guest frontends

This document specifies the front half of the ruvm dynamic binary translator: the intermediate representation that every guest frontend emits and every host backend consumes, the build-time decoder generator, the per-block translator loop, condition code handling, vector and floating point lowering, exception state recovery, I/O and icount rules, the tier-1 and tier-2 optimizers, and memory model fence placement. Document 08 covers everything after the IR: instruction selection, register allocation, the code cache, the TB cache, the softmmu TLB, MTTCG, plugins, and the performance plan. Document 09 covers the individual guest ISAs. The reference throughout is QEMU 11.1.0, and file names such as tcg/optimize.c refer to that tree.

## Scope and crate split

The JIT is split so that the parts with no QEMU-derived logic can carry the permissive license required by document 03 and the canon.

| Crate | Contents | License |
|---|---|---|
| ruvm-jit (IR core) | IR types, builder, verifier, tier-1 optimizer, liveness, tier-2 region SSA and passes, fence placement engine | MIT OR Apache-2.0 |
| ruvm-jit (runtime modules) | TB cache, jump cache, softmmu TLB runtime, cpu_exec loop, helpers shared by all targets | GPL-2.0-or-later (ported from accel/tcg) |
| ruvm-decode | decodetree-compatible generator, run from build.rs | MIT OR Apache-2.0 (the generator), generated code inherits the target crate license |
| ruvm-softfloat | bit exact port of fpu/softfloat | GPL-2.0-or-later |
| ruvm-target-<arch> | decoders (.decode files copied from QEMU), translate functions, helpers, lazy flag rules | GPL-2.0-or-later |
| ruvm-jit-<host> | backends, document 08 | GPL-2.0-or-later where ported from tcg/<host>, otherwise dual |

The runtime modules live in the same crate as the IR core but behind a `runtime` feature whose license is GPL; `cargo xtask provenance` checks the feature split, and in practice we publish them as `ruvm-jit-core` (permissive) and `ruvm-jit` (GPL) on crates.io. This is a new decision: the canon lists "ruvm-jit IR core" as permissive, and the cleanest way to honor that is two crates rather than one crate with a license-changing feature. Document 24 should record the name ruvm-jit-core.

## Why a new IR

### Why not reuse TCG ops directly

The obvious path is to port tcg/tcg-op.c, tcg/optimize.c and tcg/tcg.c line by line and keep TCG's op set. We reuse TCG's semantics almost everywhere (MemOp, barrier bits, helper flags, the gvec expansion model), because every guest frontend in QEMU is written against those semantics and we are porting those frontends. We do not reuse TCG's representation, for five reasons.

1. TCG ops are a doubly linked list of `TCGOp` with up to `TCG_MAX_TEMPS` (512) temporaries per TB and `TCG_MAX_INSNS` (512) guest instructions. Temporaries have kinds (EBB, TB, global, fixed, const) with liveness computed by `liveness_pass_1` and a second pass (`liveness_pass_2`) that makes indirect globals direct. That model is designed around one extended basic block at a time. Tier 2 in ruvm needs the same ops to live in a multi-block region with real SSA, phis, and dominance. Grafting SSA onto TCG temporaries means two representations anyway.
2. TCG's value types are I32, I64, I128 (only for a few ops), and V64/V128/V256. Flags and guest registers are memory-backed globals in `CPUArchState`. ruvm wants guest registers as first-class IR "slots" so tier 2 can promote them to SSA values across blocks without pattern-matching loads and stores of `env` offsets.
3. tcg/optimize.c (about 85 `fold_*` functions over a known-bits lattice of `z_mask`, `o_mask`, `s_mask` per temp) is good and we port its rules, but it runs over a list without use lists or def pointers, which both tiers need.
4. TCG bakes host capabilities into the op stream at emission time (`TCG_TARGET_HAS_*`, tcg/tcg-has.h). ruvm IR stays host independent until lowering, so it can be cached for tier 2 and looks the same to plugins on every host.
5. Memory ordering. Risotto (Gouicem et al., ASPLOS 2023) formalized the TCG IR memory model and showed that QEMU's x86 to Arm path emits a fence before every guest access (`tcg_gen_req_mo` in tcg/tcg-op-ldst.c), which both over-fences and prevents fence merging, and that some of QEMU's transformations are incorrect in the presence of its `Fmr` fences. We want fences as explicit IR ops placed after mapping, with a verified scheme, rather than as a side effect of each load and store helper.

The cost of this decision is that every QEMU frontend must be ported against a Rust API that looks like `tcg_gen_*` but builds ruvm IR. Document 09 shows that the API is close enough that the port is mechanical: `tcg_gen_add_i64(d, a, b)` becomes `b.add(I64, a, b)` returning a value, and the global register file becomes slot reads and writes.

### Why not Cranelift or LLVM

Both were evaluated as the tier-1 and the tier-2 backend. Neither is acceptable for tier 1, and LLVM is rejected for tier 2 as well.

Compile latency is the reason. A DBT compiles a large amount of code that runs a few times. The numbers from published work:

| Data point | Source |
|---|---|
| LLVM-based Instrew DBT: on SPEC CPU2017 602.gcc "around 55% of its time is spent on translating", and that benchmark sees almost no gain over QEMU | Engelke, Okwieka, Schulz, "Efficient LLVM-Based Dynamic Binary Translation", VEE 2021 |
| Same paper: x86-64 to x86-64 mean overhead 53% for Instrew, 124% for HQEMU, 716% for QEMU; RISC-V to x86-64 integer overhead 76% Instrew vs 202% QEMU | same |
| HQEMU runs LLVM on separate cores to hide compile time and reaches 2.4x over QEMU on SPEC CINT2006 and 4x on CFP2006 for x86 to x86-64 | Hong et al., CGO 2012 |
| Cranelift compiles only 20 to 35% faster than unoptimized LLVM; a single-pass compiler is 16x faster than Cranelift at similar code quality | Engelke and Schwarz, "Compile-Time Analysis of Compiler Frameworks for Query Compilation", CGO 2024 |
| TPDE, a single-pass SSA backend, compiles LLVM IR 8 to 24x faster than LLVM -O0 with similar run time; 4.27x faster than Cranelift on Wasmtime benchmarks, and 37% of Cranelift's time there goes to building CLIF | Schwarz, Kamm, Engelke, arXiv 2505.22610 (2025) |
| Copy-and-patch compiles about two orders of magnitude faster than LLVM -O0 and 4.9 to 6.5x faster than V8 Liftoff | Xu and Kjolstad, OOPSLA 2021 |

TCG itself is a single-pass, local-allocation code generator in the same class as the fast baselines above. Replacing it with Cranelift would make tier 1 slower to compile by roughly an order of magnitude for workloads like boot, compilers, and shells where translation dominates, which is exactly where QEMU TCG is already weakest. It would also add a second IR translation step (guest IR to CLIF) that TPDE's measurement shows costs about a third of total compile time.

Tier 2 compiles only hot regions on a background thread, so Cranelift would be workable there. We still do not use it, for three reasons. First, Cranelift has no concept of a guest PC map, softmmu memory ops or inline TLB fast paths, so our most important ops would become opaque calls or CLIF sequences the optimizer cannot see through. Second, the tier-2 wins we care about (flag liveness, register promotion, load/store forwarding, fence merging) are domain-specific passes on our IR. Third, owning the register allocator lets tier 2 and tier 1 share the same calling convention for `env`, the TLB, and helper calls, so a tier-2 region can chain directly to a tier-1 block and back without an adapter. LLVM is rejected outright: HQEMU and Instrew show both the code quality and the compile cost, and a large C++ dependency with its own release cadence is a poor fit for a Rust project that builds for 19 guest ISAs.

A Cranelift tier-2 backend is kept as an experiment behind a cargo feature (`jit-tier2-cranelift`) so that we can measure rather than argue. It is not in the product.

Copy-and-patch is used, narrowly, in the interpreter backend (document 08) and for helper thunks, not as the tier-1 code generator. The reason is that copy-and-patch stencils are compiled by clang for one host, while tier 1 needs per-op constraint handling (the `C_O1_I2(r, r, ri)` style constraints of tcg-target-con-set.h) and the inline TLB sequence, which stencils handle poorly.

## IR design

### Shape

The IR is a list of blocks, each a vector of instructions, in a per-translation arena. Each instruction defines at most two values (two for ops like `add2`, `mulu2`, and 128-bit loads split on hosts without 128-bit registers). Values are numbered densely per translation, so side tables are plain `Vec`s indexed by value id. Tier 1 translations are a single extended basic block plus out-of-line cold blocks (slow paths, exception exits), matching what TCG calls an EBB; tier 2 regions are general CFGs with phis. The representation is the same in both tiers; tier 1 simply never creates phis.

```rust
pub struct Func {
    pub blocks: Vec<Block>,          // block 0 is the entry
    pub insts: Vec<Inst>,            // arena, blocks hold ranges or index lists
    pub values: Vec<ValueData>,      // type, def site, use count
    pub slots: SlotTable,            // guest state slots declared by the target
    pub consts: ConstPool,           // i64/i128/vector constants, interned
    pub pcmap: PcMap,                // guest insn boundaries, see side tables
    pub flags: FuncFlags,            // cflags equivalent: parallel, icount, noirq, pcrel, single-step
}
pub struct Inst { pub op: Opcode, pub ty: Type, pub args: ArgList, pub imm: u64, pub aux: u32 }
pub enum Type { I32, I64, I128, V64, V128, V256, Flags, Mem }
```

`Inst` is 24 bytes with `ArgList` holding up to three inline value ids and spilling to the arena beyond that. With TCG's own limits of 512 guest instructions and 512 temporaries per TB as the ceiling, a tier-1 translation fits in a few tens of kilobytes of arena, and the arena is reset, not freed, between translations.

### Types

`I32` and `I64` are the integer types, with the same rule as TCG: an `I32` op on a 64-bit host operates on the low half and the high half is undefined unless an explicit extension op follows. `I128` exists for 128-bit guest memory ops, 128-bit CAS, and a small set of arithmetic ops (`add`, `sub`, `extract`, `concat`) that tier 1 lowers to register pairs. `V64`, `V128`, `V256` are vector registers with no element type; element size is an operand of each vector op, exactly as in tcg-op-vec.c. `Flags` is used only by lazy condition code ops (see below) and never reaches a backend except as an optional host-flags hint. `Mem` is a token type threaded through memory ops in tier 2 to express ordering; tier 1 does not materialize it because program order of the op list is the memory order.

Guest pointer width is not a type. Every guest memory op carries its address as I32 or I64 depending on the target's address size and whether the vCPU is in a 32-bit mode, matching `s->addr_type` in tcg.c.

### Guest state slots

A target declares its architectural state as slots: `slot!(RAX, I64, offset_of!(X86State, regs[0]))`. Reading and writing a slot are IR ops (`slot_ld`, `slot_st`), not generic `env` loads. Slots have attributes: `Indirect(base_slot)` for register windows (SPARC) and banked registers where the base pointer is itself state, `Volatile` for state that helpers may change behind the translator's back (the helper side-effect flags decide when it must be reloaded), and `Pc` for the program counter slot, which the translator treats specially for PC-relative translation (the `CF_PCREL` model). Generic `env` loads and stores (`ld_env`, `st_env`) remain for state that is not worth a slot, such as rarely used system registers.

Tier 1 treats slots as TCG treats globals: cached in a host register within a block, synced to memory at block exit, before helpers that may read globals, and before any op that may fault. Tier 2 promotes slots to SSA values across the region.

### Opcode set

The op set is TCG's, renamed for Rust and trimmed where TCG has historical duplicates. Integer: `mov`, `add`, `sub`, `neg`, `mul`, `muluh`, `mulsh`, `mulu2`, `muls2`, `add2`, `sub2`, `divs`, `divu`, `rems`, `remu`, `and`, `or`, `xor`, `andc`, `orc`, `eqv`, `nand`, `nor`, `not`, `shl`, `shr`, `sar`, `rotl`, `rotr`, `extract`, `sextract`, `deposit`, `extract2`, `bswap16/32/64` (with the same `TCG_BSWAP_IZ/OZ/OS` flag semantics), `clz`, `ctz`, `ctpop`, `setcond`, `negsetcond`, `movcond`, `brcond`, `ext_i32_i64`, `extu_i32_i64`, `extrl`, `extrh`. Conditions are the TCG set including `TSTEQ` and `TSTNE`. Every op has a defined result for every input: `divs` of `INT_MIN` by `-1` and division by zero are illegal in the IR (frontends must emit the guard, as QEMU frontends already do) so backends never need to handle a host trap.

Control: `br`, `brcond`, `goto_tb(slot)`, `exit_tb(code)`, `lookup_and_goto_ptr`, `trap(kind)`. `goto_tb` has two slots per block, like `tb->jmp_target_addr[2]`. `exit_tb` codes follow TB_EXIT_* in include/tcg/tcg.h.

Memory: `qemu_ld`, `qemu_st`, `qemu_ld128`, `qemu_st128`, atomic ops (below), `mb(bar)`. Host memory ops on `env` are separate and never go through the TLB.

Calls: `call(helper, args)` with a `HelperInfo` describing ABI types and side-effect flags.

Guest bookkeeping: `insn_start(pc, data...)` marks a guest instruction boundary and carries up to three target words (the `TARGET_INSN_START_EXTRA_WORDS` equivalent); `plugin_cb` placeholders for plugin instrumentation (document 08); `icount_dec` and `io_start` for icount (below).

### Memory ops and MemOp

Every guest memory op carries a `MemOpIdx` exactly compatible with QEMU's `make_memop_idx(memop, mmu_idx)`, because document 08's softmmu and the plugin ABI expose it (`qemu_plugin_meminfo_t` encodes it). The MemOp bits are the ones in include/exec/memop.h:

| Field | Values | Meaning |
|---|---|---|
| size | MO_8, MO_16, MO_32, MO_64, MO_128 | access size |
| sign | MO_SIGN | sign extend the loaded value |
| endian | MO_BSWAP (host-relative), MO_LE, MO_BE | byte order |
| align | MO_UNALN, MO_ALIGN_2 to MO_ALIGN_64, MO_ALIGN | required alignment, fault via `do_unaligned_access` when violated |
| atom | MO_ATOM_IFALIGN, MO_ATOM_IFALIGN_PAIR, MO_ATOM_WITHIN16, MO_ATOM_WITHIN16_PAIR, MO_ATOM_SUBALIGN, MO_ATOM_NONE | single-copy atomicity required |

The atomicity field matters more in ruvm than in QEMU, because ruvm is actually going to run strong-on-weak with MTTCG on by default. The rule, taken from accel/tcg/ldst_atomicity.c.inc and required for correctness by Arancini's mixed-size result (splitting a 16-bit store into two 8-bit stores admits outcomes x86 forbids), is: a backend may split an access only when the MemOp's atom field permits it for the actual alignment at run time. `MO_ATOM_IFALIGN` means a naturally aligned access must be a single host access; `MO_ATOM_WITHIN16` (x86's rule for SSE and for any access not crossing a 16-byte boundary on CPUs with AVX) means an access that does not cross 16 bytes must be single-copy atomic even if misaligned; `MO_ATOM_SUBALIGN` (used by s390x and some Arm cases) means atomic in pieces of the natural alignment of the address. When the host cannot provide the requested atomicity inline (for example an unaligned 8-byte access within 16 bytes on a host without FEAT_LSE2), the op takes the slow path, and in a parallel TB the slow path may require `exit_atomic` and serial re-execution (document 08).

### Atomic ops

Atomics are IR ops, not helper calls, so tier 2 can reason about them and backends can inline them.

| Op | Semantics | Notes |
|---|---|---|
| `atomic_cmpxchg(addr, cmp, new, oi)` | returns old value | I32, I64, I128 |
| `atomic_xchg` | returns old | |
| `atomic_fetch_{add,and,or,xor,smin,smax,umin,umax}` | returns old | |
| `atomic_{add,and,or,xor,...}_fetch` | returns new | |
| `ll(addr, oi)` / `sc(addr, val, oi)` | reservation pair for guests whose frontends emulate LL/SC natively | only used when the host path in document 08 permits |

All atomics carry an ordering attribute: `Relaxed`, `Acquire`, `Release`, `AcqRel`, or `SeqCst`. In QEMU, atomics called from translated code in a parallel TB are sequentially consistent helpers built from `qatomic_*` in accel/tcg/atomic_template.h, and when the TB is not parallel (`CF_PARALLEL` clear) frontends emit plain load-op-store. ruvm keeps that split: frontends call `b.atomic_*` unconditionally and the IR builder lowers to a non-atomic sequence when the translation's `parallel` flag is clear, which is what `tcg_gen_atomic_*` does in tcg-op-ldst.c. This keeps single-threaded guests (and icount, which is serial) free of atomic overhead.

The ordering attribute is filled in by the frontend from guest semantics and then rewritten by the memory model pass: an x86 `lock` prefixed instruction produces `SeqCst`, an Arm `ldaxr/stlxr` pair produces acquire and release, a RISC-V `amoadd.w.aq` produces `Acquire`. The attribute is what the fence placement tables below key on.

### Barriers

`mb(bar)` carries the TCG barrier encoding from include/tcg/tcg-mo.h: the four ordering bits `TCG_MO_LD_LD`, `TCG_MO_ST_LD`, `TCG_MO_LD_ST`, `TCG_MO_ST_ST` plus the kind `TCG_BAR_LDAQ`, `TCG_BAR_STRL`, `TCG_BAR_SC`. We add a provenance annotation that does not exist in TCG:

```rust
pub enum FenceOrigin { Guest, Mapping(MappingRule), Helper }
```

`Guest` fences come from explicit guest instructions (x86 `mfence`, Arm `dmb`, RISC-V `fence`) and are never removed except when merged with an adjacent fence that subsumes them. `Mapping` fences are inserted by the memory model pass to implement strong-on-weak translation and carry the rule that produced them, so the merge and elimination passes know which proof obligation each fence discharges. `Helper` fences surround helpers that access guest memory outside the IR's view.

### Helper calls and side effects

A helper is a Rust `extern "C"` function registered with a `HelperInfo`: argument and return types (I32, I64, I128, pointer, env) and flags. The flags are TCG's, with the same bit values so ported frontend code reads the same:

| Flag | TCG name | Meaning for the optimizer |
|---|---|---|
| NO_READ_GLOBALS | TCG_CALL_NO_READ_GLOBALS (TCG_CALL_NO_RWG) | slot values need not be synced to memory before the call |
| NO_WRITE_GLOBALS | TCG_CALL_NO_WRITE_GLOBALS (TCG_CALL_NO_WG) | slot values cached in registers stay valid after the call |
| NO_SIDE_EFFECTS | TCG_CALL_NO_SIDE_EFFECTS (TCG_CALL_NO_SE) | the call may be deleted if its result is unused |
| NO_RETURN | TCG_CALL_NO_RETURN | the call never returns normally (raises an exception via unwinding to the cpu loop) |

ruvm adds three flags. `MAY_FAULT` says the helper can raise a guest exception, so the translator must have a precise-state point (see side tables) registered at the call's return address; QEMU implicitly assumes every helper may fault and relies on `GETPC()`. `READS_MEM` and `WRITES_MEM` say the helper touches guest memory, which fixes its position relative to memory ops in tier 2 and determines whether a `Helper` fence is needed. Helpers without either flag are pure register computations and can be scheduled freely. A debug build checks the declarations by sampling: it snapshots slot memory around calls and flags undeclared writes.

### Exceptions and control flow out of a block

Guest exceptions leave translated code in one of three ways. A helper raises (`raise_exception` style helpers are `NO_RETURN`). A memory op faults in the TLB slow path. Or the IR emits `trap(kind)` for inline checks such as x86 `into` or Arm alignment checks, which lowers to a conditional branch to a cold stub that calls the raise helper. In all three cases the unwinding mechanism is the same as QEMU's `cpu_loop_exit` (a longjmp in C, in ruvm a controlled unwind implemented as a non-local return through a saved host stack pointer in the cpu loop, because Rust panics are too slow and must not cross the `extern "C"` boundary of generated code). Before unwinding, the runtime restores guest state from the side table using the host return address, as described below.

## Decoder generation from decodetree files

QEMU describes most modern guest encodings in decodetree files (target/arm/tcg/a64.decode, target/riscv/insn32.decode, target/loongarch/insns.decode, parts of target/i386 via its own table decoder, and others), processed at build time by scripts/decodetree.py and documented in docs/devel/decodetree.rst. ruvm consumes the same `.decode` files unmodified. This is a compatibility decision, not a convenience: the files encode which bit patterns are valid, which overlap, and in which order overlapping patterns are tried, and any drift from QEMU shows up as differently-handled undefined encodings, which the tcg tests and our differential fuzzer (document 22) catch but which are tedious to chase.

ruvm-decode is a Rust reimplementation of decodetree.py, run from each target crate's build.rs. It parses the full syntax (fields with `!function=` transforms, argument sets, formats, patterns, overlapping `{ }` groups tried in order, non-overlapping `[ ]` groups checked for disjointness) and the `--insnwidth`, `--varinsnwidth`, and `--static-decode` options.

Output is a Rust module with one argument struct per argument set and a `decode(ctx, insn) -> bool` function that calls `trans_<name>(ctx, &args) -> bool` methods on a trait the target implements. Returning false from a `trans_` function means "not handled here, keep trying", matching decodetree's semantics for overlapping groups. The decision tree is built the way decodetree.py builds it (partition on the bits fixed across all patterns in a group, recursively) and emitted as nested `match` on masked values.

Two checks run at build time that decodetree.py does not have. First, every `trans_` name referenced in the file must exist on the trait, and every trait method must be referenced, so a missing implementation is a compile error rather than a silent undefined instruction. Second, the generator emits a table of (mask, value, name) for every pattern that the differential fuzzer uses to generate instruction streams that cover every pattern.

x86 does not use decodetree for its whole decoder in QEMU; target/i386/tcg/decode-new.c.inc is a table-driven decoder with opcode maps and operand types. ruvm-target-x86 ports that table decoder directly (document 09), because translating the tables into decodetree syntax would be a rewrite with its own bugs.

## Translator loop

The translator loop is the Rust equivalent of `translator_loop` in accel/tcg/translator.c, with the same hook structure as `TranslatorOps` (include/exec/translator.h):

```rust
pub trait Translator {
    type Ctx: DisasContext;
    fn init_disas_context(&self, ctx: &mut Self::Ctx, cpu: &CpuView);
    fn tb_start(&self, ctx: &mut Self::Ctx, b: &mut Builder);
    fn insn_start(&self, ctx: &mut Self::Ctx, b: &mut Builder);     // emits insn_start with pc and extra words
    fn translate_insn(&self, ctx: &mut Self::Ctx, b: &mut Builder); // decodes one guest insn, advances pc_next
    fn tb_stop(&self, ctx: &mut Self::Ctx, b: &mut Builder);        // emits goto_tb / lookup_and_goto_ptr / exit_tb
    fn disas_log(&self, ctx: &Self::Ctx, out: &mut dyn Write) -> bool { false }
}
pub enum DisasJump { Next, TooMany, NoReturn, Target(u32) } // DISAS_NEXT, DISAS_TOO_MANY, DISAS_NORETURN, DISAS_TARGET_n
```

The loop, per block:

1. Compute the lookup key: `(pc, cs_base, flags, cflags)` from the target's `get_tb_cpu_state`, same fields as `TCGTBCPUState`.
2. `init_disas_context`, then `gen_tb_start`: if icount or exit requests are enabled, emit the load of `icount_decr.u32` (the combined exit-request and icount budget word in `CPUNegativeOffsetState`) and a branch to the exit path if it is negative. This is the only interrupt check in a block; document 08 explains why it is enough.
3. Loop: `insn_start`, plugin insn hook placeholder, `translate_insn`. After each instruction check termination: `is_jmp != Next`, number of instructions reached `max_insns` (from `CF_COUNT_MASK` or the global limit of 512), the next instruction would start on a different guest page than the first (except the single permitted page crossing, see below), the IR arena is near its limit, or single-step is active.
4. `tb_stop`, `gen_tb_end` (icount decrement fixup), plugin block hooks, finalize the PC map.

Page crossing follows QEMU exactly. A block may contain instructions from at most two guest pages, and the second page is only allowed when an instruction straddles the boundary or the block legitimately continues onto the next page; the TB records both physical pages (`page_addr[0..2]` in QEMU's `TranslationBlock`) so that invalidating either invalidates the block. Instruction fetch goes through `translator_ld*` equivalents that record which host page each byte came from and, in plugin builds, which bytes form each instruction (for `qemu_plugin_insn_data`). If a fetch faults on the second page, translation stops before that instruction and the block ends, so the fault is raised when the next block starts, with precise state.

The block size limit is a policy knob. QEMU uses up to 512 guest instructions and one page. ruvm tier 1 uses the same limits, because they bound invalidation cost and the jump cache already makes block-to-block transitions cheap once chained. Tier 2 regions (below) span pages.

`CF_PCREL` is the default for every target that supports it in QEMU (x86, Arm, RISC-V, LoongArch, and others listed in document 09). With PC-relative translation the generated code does not embed the guest virtual PC; it computes it from the PC slot at the block entry. This allows the same translation to be reused for a physical page mapped at several virtual addresses (common with Linux's kernel direct map and with user-mode `mmap` aliases) and is required by our TB cache design where the primary key is the physical address.

## Lazy condition codes

Flags are the single largest source of redundant work in a DBT for flag-heavy guests. QEMU's frontends already do lazy flags (target/i386 with `CC_OP` and `cc_dst/cc_src/cc_src2`, target/arm computing NZCV into four separate globals `NF`, `ZF`, `CF`, `VF` in a form that makes each cheap, target/s390x with `cc_op`). ruvm keeps each target's QEMU representation, because it is observable through gdbstub, migration (the VMState of the CPU contains `cc_op` and friends for x86), and helpers, and adds a generic flags layer so tier 1 and tier 2 can eliminate flag work the way Rosetta 2 and FEX do.

### Representation in IR

A flag-setting guest instruction emits its result and a `flags_def(kind, a, b, result)` op of type `Flags` that names the operation (`AddI32`, `SubI64`, `LogicI8`, `ShlI32`, and so on, the same list as x86's `CC_OP_*` extended with Arm and s390x forms). Consumers emit `flags_use(flags, cond)` producing an I32 truth value, or `flags_materialize(flags, which)` producing the architectural representation for a specific flag or for the whole flag word. Stores of the architectural representation to state slots are also emitted by the frontend, as they are in QEMU, but they are marked `lazy` so dead-store elimination may remove them when a later `flags_def` overwrites them before any possible observer.

Observers are: a later `flags_use`, a helper without NO_READ_GLOBALS, block exit (the flags escape to the next block), and any op that can fault (flags must be architecturally correct at a precise exception point). The last rule is the one that makes flag elimination hard. ruvm handles it the same way for flags as for all state: the side table can record that the flags at a faulting point are "the result of `flags_def` N applied to values held in host locations L1, L2", and the restore routine recomputes them. That way a faulting load between a `cmp` and a `jcc` does not force the `cmp` to materialize flags in the fast path.

### Per architecture rules

x86 guest. `CC_OP` state is kept, but within a block the frontend tracks the pending flags definition statically (QEMU already does this in `DisasContext.cc_op` with `set_cc_op` and `gen_update_cc_op`). A `jcc` or `setcc` after a `cmp` or `sub` becomes one `brcond`/`setcond` on the operands, which QEMU also does via `gen_prepare_cc`. ruvm's addition is cross-block: tier 2 computes flag liveness over the region and drops the `cc_op`, `cc_src`, `cc_dst` stores on paths where the next definition precedes any observer. PF and AF are computed only on demand from `cc_dst` and `cc_src` (as QEMU does with `compute_all_*` helpers), never eagerly. On an aarch64 host the backend may carry CF, ZF, SF, OF in host NZCV between a definition and a use within a block (host-flags hint), with the carry inversion Rosetta 2 uses: Arm computes subtraction carry as NOT borrow, so the canonical host form after a subtraction is inverted relative to x86 and `cfinv` (FEAT_FlagM) fixes it only when the flag escapes. `rmif`, `setf8`, and `setf16` (FEAT_FlagM) are used when present for shifts and 8/16-bit operations; `axflag`/`xaflag` (FEAT_FlagM2) for `ucomis*` results. "ARMing x86 Games" (Yen et al., MobiSys 2025) reports up to 18% from speculatively substituting Arm hardware flags for x86 flags; our host-flags hint is the non-speculative subset, and the speculative part is listed in document 25.

Arm guest. QEMU's A64 translator keeps `NF` as a value whose sign bit is N, `ZF` as a value that is zero iff Z is set, `CF` as 0 or 1, and `VF` as a value whose sign bit is V. This is already lazy for Z and N. ruvm keeps it and adds the flags_def/flags_use layer on top so that an `adds`+`b.cond` pair on an x86 host lowers to `add` plus `jcc` with host flags, and on an aarch64 host to the same instruction pair.

RISC-V, MIPS, LoongArch, Alpha, and other flagless guests have no condition codes; nothing to do.

s390x, PowerPC, SPARC, m68k, and other flag guests keep their QEMU representation (`cc_op` style state, PowerPC's eight CR fields as eight `Flags` values) behind the same layer.

## Vector ops and host SIMD mapping

Guest SIMD (SSE through AVX2 and AVX-512 on x86, Neon, SVE, and SME on Arm, RVV on RISC-V, VSX on PowerPC, the s390x vector facility, LSX/LASX on LoongArch) is translated through two layers, as in QEMU.

The upper layer is the gvec API, a port of tcg/tcg-op-gvec.c. A gvec op names a destination and sources as byte offsets into `env`, an operation size `oprsz` and a maximum size `maxsz` (bytes beyond `oprsz` up to `maxsz` are zeroed, which is how x86 VEX zeroing of upper lanes and SVE predicate-width handling are expressed), and an element size. The API chooses an expansion: inline vector ops of the widest host vector type that divides `oprsz`, inline 64-bit integer ops for small sizes where that is cheaper, or an out-of-line helper taking a `simd_desc` word (oprsz, maxsz, and a data field packed as in `simd_desc()`), exactly as `GVecGen2`, `GVecGen3`, `GVecGen4` do in QEMU. The expansion table for each operation lists the vector op list it requires (`vecop_list`), and the expansion is chosen by querying the backend's capability for each op at the requested element size, the ruvm equivalent of `tcg_can_emit_vecop_list`.

The lower layer is the vector IR ops of tcg-op-vec.c: `dup`, `ld_vec`, `st_vec`, `add`, `sub`, `mul`, `neg`, `abs`, saturating add and subtract, `smin/smax/umin/umax`, logical ops, shifts by immediate, by scalar, and by vector, rotates, `cmp`, `bitsel`, and `cmpsel`. Types are V64, V128, V256; element sizes MO_8 to MO_64.

ruvm differs from QEMU in one place: guest vector registers can be slots. QEMU gvec always works on `env` memory, which means every guest SIMD instruction loads its operands from `env` and stores its result back. For a sequence like `movdqa; paddd; pxor; movdqa` that is 3 to 4 memory round trips per instruction. In ruvm the target declares its vector registers as V128 or V256 slots and gvec ops whose offsets exactly cover a slot are rewritten to slot reads and writes, which tier 1's local allocator then keeps in host vector registers within a block. Offsets that do not align with a slot (partial lane updates, element inserts) fall back to `env` memory, after syncing the affected slot. This is where much of the SIMD speedup over TCG comes from (quantified in document 08's performance plan) and is the same approach Rosetta 2 and FEX take by mapping SSE registers directly to host vector registers.

Host mapping:

| Guest vector width | x86-64 host | aarch64 host | riscv64 host |
|---|---|---|---|
| 64-bit (MMX, Neon D) | SSE2 xmm, low half | Neon D | RVV with vl for 64 bits, else scalar pairs |
| 128-bit (SSE, Neon Q, LSX, VSX) | SSE2 to SSE4.1, AVX encodings when available (three-operand) | Neon Q | RVV with VLEN at least 128 |
| 256-bit (AVX2, LASX) | AVX2 ymm | two Neon Q registers | RVV LMUL=2 at VLEN 128, LMUL=1 at VLEN 256 |
| 512-bit (AVX-512) | AVX-512 zmm when present, else two ymm | four Q registers or SVE at VL 512 | RVV LMUL grouping |
| scalable (SVE, RVV guest) | helper expansion for predicated ops, inline for unpredicated at the current VL | SVE when host VL is at least guest VL, else Neon plus helpers | RVV |

Host features are detected once at startup (cpuid, `getauxval(AT_HWCAP)`, `sysctl hw.optional` on macOS, `riscv_hwprobe`) and published as a capability bitset that the gvec expansion queries. Where QEMU's tcg/riscv64 backend supports vector ops only when RVV is present, so does ours; without RVV a riscv64 host expands vector ops to 64-bit scalar sequences or helpers.

Predicated operations (SVE, AVX-512 opmasks, RVV masks) are inline only when the predicate is all-true at translation time; otherwise they go to helpers, as in QEMU.

## Floating point

### Softfloat

ruvm-softfloat is a line-by-line port of QEMU's fpu/softfloat.c, fpu/softfloat-parts.c.inc, and fpu/softfloat-specialize.c.inc into Rust, with the same `float_status` fields: rounding mode, exception flags, `floatx80_rounding_precision`, `flush_to_zero`, `flush_inputs_to_zero`, `default_nan_mode`, `snan_bit_is_one`, the NaN propagation rules (`float_2nan_prop_rule`, `float_3nan_prop_rule`, `float_infzeronan_rule`), the default NaN pattern, and tininess detection before or after rounding. Every target selects its rules at CPU reset as QEMU does. Bit exactness is checked by running QEMU's tests/fp suite (the TestFloat and berkeley-softfloat-3 based `fp-test`) against both implementations, plus a differential fuzzer that feeds the same random operands and status to QEMU's C softfloat (linked into the test binary) and ours.

### When host FP is used

QEMU has a hardfloat fast path in fpu/softfloat.c. It uses the host FPU for add, sub, mul, div, fma, and sqrt on float32 and float64 when `can_use_fpu()` holds: the inexact flag is already set in the guest status and the rounding mode is round-to-nearest-even. The inputs must be zero or normal (`f32_is_zon2` and friends), and after the host operation the result is checked: an infinite result sets overflow, and a result with magnitude at or below `FLT_MIN` (or `DBL_MIN`) falls back to the soft path to get underflow and tininess right. The reasoning is that with inexact already set, the only flags the host operation could need to raise are overflow, underflow, and invalid, and the input and output checks cover them. It is disabled on some hosts via `QEMU_NO_HARDFLOAT`.

ruvm keeps that path as the helper-level default and adds an inline path, emitted directly in translated code, under stricter rules. An FP op may be lowered to a host instruction inline when all of the following hold for the translation, which the frontend proves from the TB flags or guards at block entry:

1. The rounding mode is known at translation time and the host can encode it statically or it is the host's current mode. x86 SSE encodes a static mode in `roundss`-style instructions only; for arithmetic ruvm requires the guest mode to be round-to-nearest-even, which it is for effectively all user code.
2. The guest's accumulated inexact flag is already set, or the guest does not observe inexact before the next point where the block re-checks it. In practice we require it to be set, exactly as QEMU does, and the flag word is part of the TB lookup key only through a single bit ("inexact already set and RNE") so that blocks do not multiply.
3. Flush-to-zero and denormals-are-zero settings match what the host is configured with for translated code. Translated code runs with the host FP control register set to RNE, no FTZ, no DAZ, all exceptions masked; a guest with FTZ enabled uses helpers.
4. NaN results are either impossible for the operation given input checks, or the host's NaN result matches the guest's rule. For add, sub, mul, div with non-NaN inputs, the only NaN outputs are invalid operations (inf minus inf, zero times inf, zero divided by zero), and the default NaN differs between architectures: x86 produces the "real indefinite" negative quiet NaN (sign set), Arm with default NaN mode produces a positive quiet NaN, RISC-V always produces the canonical NaN 0x7fc00000. The inline path therefore checks that the result is not NaN, and on NaN takes the helper path, which recomputes with softfloat and sets flags. When inputs may be NaN the inline path is not used at all, because propagation rules (which operand's payload wins, whether signaling NaNs are quieted) differ.
5. The operation is one where IEEE 754 fully specifies the result: add, sub, mul, div, sqrt, and fused multiply-add when the host has a true single-rounding FMA. Conversions, min/max (x86 `minss` is not IEEE minNum, and Arm `fmin` vs `fminnm` differ), reciprocal estimates, and x87 80-bit operations always use softfloat helpers. x87 is never done in host FP; floatx80 goes through softfloat, and document 09 describes the targeted fast path for 64-bit-precision x87 control words.
6. Tininess and underflow: after the inline op, a check that the result magnitude is above the smallest normal (or exactly zero from non-subnormal operands, which cannot underflow) keeps us on the fast path; otherwise the helper recomputes. This is the QEMU hardfloat post-check, done inline.

The checks add a few host instructions per op (classify inputs, operate, classify output), far cheaper than a call into softfloat. Exception flags are never read from the host FP status register in translated code, because reading and clearing MXCSR or FPSR is serializing on some cores.

Guest FP exception trapping (x86 unmasked exceptions, PowerPC FE0/FE1, Arm trapped exceptions) forces the helper path for every FP op in the affected TBs; a TB flag bit records whether any exception is unmasked.

## Precise exception state recovery via side tables

Every point in translated code where control can leave for the exception path must be able to reconstruct guest architectural state as of the start of the faulting guest instruction (or after it, for traps that report the next PC). QEMU does this with `insn_start` ops whose arguments (guest PC plus target-specific extra words such as x86's `cc_op` or Arm's syndrome bits) are encoded after the TB's host code as a table of sleb128 deltas, one row per guest instruction, with the host code end offset of that instruction; `cpu_unwind_data_from_tb` in accel/tcg/translate-all.c walks the table linearly from the start to find the row for a given host return address, and the target's `restore_state_to_opc` applies the words. Because tier 1 keeps all guest registers synced to `env` before any op that may fault, the PC and the extra words are the only state that needs restoring.

ruvm keeps the same scheme for tier 1, with two changes. First, the side table is stored in a separate region of the code buffer rather than immediately after the code, so the executable pages contain only code and the table pages can be read-only data (this matters for W^X and for i-cache density, see document 08). Second, rows are indexed by a small per-TB array of every 16th row so lookup is O(16) steps instead of O(instructions in TB).

Tier 2 needs more, because it keeps guest registers and flags in host registers across instructions and blocks. A tier-2 side table row maps a host code offset to: the guest PC, the target extra words, and for each guest slot whose memory copy is stale at that point, its location (host register, spill slot, constant, or a `flags_def` recipe). The format is a compact variant of what JavaScript and Java JITs call deopt metadata: a per-region location dictionary, and per row a bitmap of stale slots plus indices into the dictionary. The restore routine writes stale values to `env` using the register contents captured at the fault (the signal context for faulting host accesses, the saved register area for helper calls). This is what the canon means by "precise exceptions via side tables, not by restoring from re-translation": we never re-run the translator to find state, which QEMU did historically and which is incompatible with tier 2's non-deterministic register allocation.

Only ops that can leave translated code get rows: guest memory ops, `MAY_FAULT` calls, and `trap`.

## I/O and icount

MMIO accesses from translated code go through the softmmu slow path (document 08) and are safe at any point in a normal TB. icount (`-icount shift=N`, and record/replay, document 17) adds one rule: a device access must see an exact instruction count. QEMU enforces this with `can_do_io`: `translator_loop` clears it at block start and sets it only for the last instruction, frontends call `translator_io_start` before instructions that touch timers or I/O, and an MMIO access with `can_do_io` false calls `cpu_io_recompile`, which regenerates a TB that ends at the faulting instruction (with `CF_MEMI_ONLY | CF_NOIRQ` and a count of 1 or 2) and restarts it. ruvm implements the same protocol with the same observable effect, because record/replay logs are only interoperable if instruction counts match (document 17). The block prologue subtracts the block's instruction count from the 16-bit budget in `icount_decr`, and exits if it goes negative, as `gen_tb_start` does. When icount is off, none of this is emitted.

## Tier-1 optimizer

Tier 1 runs three passes over the single-EBB function, each linear in instruction count. The whole tier-1 pipeline must stay within 1.2x of QEMU TCG's translation time per guest instruction on the document 21 workloads (a budget, decided here).

1. Constant folding and copy propagation, fused into one forward pass, a port of tcg/optimize.c. Each value has a known-bits record (`z_mask`, `o_mask`, `s_mask`) and a copy-class link. The pass folds constants, applies algebraic identities (x+0, x&-1, x^x), narrows extensions whose inputs are known extended, folds `setcond`+`brcond` pairs, converts `brcond` on a constant into `br`, and rewrites uses of copies to the class representative. Knowledge of slot values is invalidated at helper calls without NO_WRITE_GLOBALS and at block boundaries. Complexity O(n) with a small constant per op; copy classes are circular lists as in QEMU, so invalidating a class is O(class size), bounded in practice by the number of live copies.
2. Dead code elimination, backward, driven by use counts. Ops with no side effects whose results have zero uses are removed and their operands' counts decremented, which cascades within the same backward sweep. O(n).
3. Liveness and sync analysis, backward, the port of `liveness_pass_1`: for each op, which operands die here and which slots must be synced to memory or can be discarded. O(n times live slots) with slots represented as bitsets, so effectively O(n) for targets with up to 64 slot words per bitset chunk.

Lazy flag stores marked by the frontend are removed in pass 2 when a later `flags_def` in the same block overwrites them with no observer between.

## Tier-2 region formation and optimization

A block's entry counter (a 16-bit decrement in the TB header, not in the fast path of chained jumps; see document 08 for where it is placed) triggers a region request when it reaches a threshold (default 4,000 executions, tunable). The request is queued to a background compiler thread pool so the vCPU never waits.

Region formation uses recorded edge profiles from tier-1 exits and chained jump counters. The region grows from the hot block along edges with at least 10% of the block's executions, following both directions of conditional branches when both are hot, up to 64 blocks or 4,096 guest instructions. Loops are the main target: a back edge to the region head closes a loop. Calls whose callee is hot and small are included with their return edge (guest call/return pairs identified by the frontend's `is_call` and `is_ret` hints). Regions may span guest pages; each page in the region is registered for SMC invalidation like any TB page.

The region is built into SSA from the cached tier-1 IR of its blocks (tier 1 keeps IR for blocks above half the threshold in an LRU cache, so tier 2 does not re-decode), with slots converted to SSA values using the standard dominance-frontier phi placement over the small CFG. Passes, in order:

1. Slot promotion: slots become SSA values; loads of slot memory disappear inside the region; stores are sunk to region exits and to side-table recipes at fault points.
2. Global value numbering and constant folding with the tier-1 known-bits lattice, over the dominator tree. O(n log n) in practice.
3. Flag liveness across blocks: a backward dataflow over `Flags` values; `flags_def` ops with no live observer are deleted and their fault-point recipes kept.
4. Load/store forwarding and redundant load elimination for guest memory, restricted by the memory model: a load may be forwarded from an earlier store or load to the same address only if no fence or atomic op with acquire semantics intervenes, the MemOp sizes match, and both are to the same TLB entry within the region (the address check is the same guest address value, not alias analysis on host pointers). This is the RAW transformation Risotto proved correct only when no `Fmr` or `Fwr` fence lies between, which is exactly why the fence mapping below avoids emitting those fence kinds.
5. TLB check hoisting: repeated guest memory ops through the same base register and page within a loop body share one TLB lookup guarded at loop entry, with a page-crossing check on the offset range. If the guard fails the region exits to tier 1.
6. Fence merging and elimination (below).
7. Linear scan register allocation over the region (document 08).

Invalidation of any page in the region, or a change in any TB flag the region was specialized on, unlinks the region and falls back to tier-1 code, which is never freed while a region depends on it.

## Memory model fence placement

### Strong guest on weak host

Canon decision: fences for strong-on-weak translation follow the verified mappings from Risotto (ASPLOS 2023) and Arancini (ASPLOS 2026). Risotto observed that QEMU inserts a fence before every load (`Fmr`, lowered to `dmb ishld` on Arm) and before every store (`Fmw`, lowered to a full `dmb ish`), that fences alone cost up to 75% of execution time on some PARSEC benchmarks (48% on average in their measurement), and that the placement both admits x86-forbidden behaviors with some compiler-generated RMW helpers and blocks fence merging. The verified schemes put a fence after loads and before stores, which allows adjacent fences to merge.

x86-64 guest, via ruvm IR, to Arm (Risotto figure 7, Arancini table 2):

| x86 | ruvm IR | aarch64 |
|---|---|---|
| load (RMOV) | `ld; F(rm)` | `ldr; dmb ishld` |
| store (WMOV) | `F(ww); st` | `dmb ishst; str` |
| locked RMW, xchg | `rmw SeqCst` | `casal` / `ldaddal` / `swpal` (FEAT_LSE), else `dmb ish; ldxr..stxr loop; dmb ish` |
| mfence | `F(mm)` | `dmb ish` |
| IR fence Frr, Frw, Frm | | `dmb ishld` |
| IR fence Fww | | `dmb ishst` |
| IR fence Fwr, Fwm, Fmr, Fmw, Fmm | | `dmb ish` |

x86-64 guest to RISC-V (Arancini table 2; the RMW instruction choice is ours, following the RISC-V ISA manual's recommended mapping for sequentially consistent RMW):

| x86 | ruvm IR | riscv64 |
|---|---|---|
| load | `ld; F(rm)` | `ld; fence r,rw` |
| store | `F(ww); st` | `fence w,w; sd` |
| locked RMW | `rmw SeqCst` | `amo<op>.aqrl`, `amocas.aqrl` with Zacas, else `lr.aqrl` / `sc.rl` loop |
| mfence | `F(mm)` | `fence rw,rw` |

On x86-64 hosts no mapping fences are needed for any guest, and only fences that include store-to-load ordering emit code: `lock orl $0,0(%rsp)`, which QEMU's tcg/x86_64 backend uses because it measured faster than `mfence`. TCG's `TCG_TARGET_DEFAULT_MO` for x86 hosts (`TCG_MO_ALL & ~TCG_MO_ST_LD`) states the same fact.

Special cases. On Apple silicon, which implements a hardware TSO mode that Rosetta 2 relies on, ruvm enables TSO for vCPU threads when the OS lets a process request it and then emits no mapping fences; this is a run-time capability probe, and without it the tables above apply. With FEAT_LRCPC, loads may use `ldapr` instead of `ldr; dmb ishld` (FEX's choice) for aligned accesses; this is an ruvm optimization not covered by the Arancini proofs and stays behind a flag until proven (document 25). Mixed-size accesses follow the MemOp atomicity rule above: an access is never split on the fast path if Arancini's mixed-size result shows the split admits new outcomes.

Weak guests (Arm, RISC-V, PowerPC) on any host need only their explicit barriers mapped to the nearest host fence that is at least as strong, as QEMU does with `tcg_out_mb`.

### Fence merging and elimination

Fences are merged only when they are adjacent, with no memory access between them, which is the merge Risotto proves sound. The common case is a load followed by a store: `ld; F(rm); F(ww); st` becomes `ld; F(rm+ww); st`, lowered on Arm to one `dmb ish` instead of `dmb ishld; dmb ishst`. A fence is never moved across a memory access to create adjacency. Arancini notes that program context such as thread-local stack accesses could remove more fences; ruvm implements that only in tier 2 for linux-user, where thread stacks are known, behind a flag that stays off until the transformation has a proof (document 25).

## References

QEMU source: tcg/tcg.c, tcg/optimize.c, tcg/tcg-op-ldst.c, tcg/tcg-op-gvec.c, accel/tcg/translator.c, accel/tcg/translate-all.c, include/exec/memop.h, include/tcg/tcg-mo.h, fpu/softfloat.c, docs/devel/decodetree.rst, docs/devel/tcg-ops.rst. Risotto: Gouicem et al., ASPLOS 2023, https://doi.org/10.1145/3567955.3567962. Arancini: Reimers et al., ASPLOS 2026, https://doi.org/10.1145/3779212.3790127. Instrew: Engelke and Schulz, VEE 2020, https://doi.org/10.1145/3381052.3381319, and Engelke, Okwieka, Schulz, VEE 2021, https://doi.org/10.1145/3453933.3454022. HQEMU: Hong et al., CGO 2012, https://doi.org/10.1145/2259016.2259030. Copy-and-patch: Xu and Kjolstad, OOPSLA 2021, https://doi.org/10.1145/3485513. Engelke and Schwarz, CGO 2024, https://home.cit.tum.de/~engelke/pubs/2403-cgo.pdf. TPDE: https://arxiv.org/abs/2505.22610. ARMing x86 Games: Yen et al., MobiSys 2025, https://doi.org/10.1145/3711875.3729163. Rosetta 2 analysis: https://dougallj.wordpress.com/2022/11/09/why-is-rosetta-2-fast/. FEX memory model notes: https://github.com/FEX-Emu/FEX/blob/main/FEXCore/docs/MemoryModelEmulation.md.
