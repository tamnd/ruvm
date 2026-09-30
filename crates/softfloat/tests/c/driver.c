/* SPDX-License-Identifier: GPL-2.0-or-later */
/*
 * The QEMU side of the ruvm-softfloat differential harness.
 *
 * This file includes QEMU's own fpu/softfloat.c, so it is built against the exact
 * code ruvm-softfloat ports (hardfloat paths included). It reads fixed size request
 * records on stdin, runs each one through QEMU, and writes a fixed size result record
 * to stdout. tests/common/mod.rs defines the same record layout and op numbers.
 *
 * Request, 80 bytes, little endian:
 *   0  u16 op (format * 64 + operation)
 *   2  u8  rounding mode        3  u8 floatx80 precision
 *   4  u8  flush_to_zero        5  u8 flush_inputs_to_zero
 *   6  u8  default_nan_mode     7  u8 tininess_before_rounding
 *   8  u8  ftz_before_rounding  9  u8 snan rule
 *   10 u8  2nan rule            11 u8 3nan rule
 *   12 u8  infzeronan rule      13 u8 floatx80 behaviour
 *   14 u8  default nan pattern  15 u8 rounding mode operand
 *   16 u16 initial flags        18 u8 rebias_overflow  19 u8 rebias_underflow
 *   20 i32 imm                  24 i32 imm2            28 u32 unused
 *   32 u64 a.lo  40 u64 a.hi  48 u64 b.lo  56 u64 b.hi  64 u64 c.lo  72 u64 c.hi
 *
 * Result, 32 bytes: u64 lo, u64 hi, u64 extra, u16 flags, 6 bytes zero.
 *
 * A request with op 0xffff produces no result; it flushes stdout, so that the Rust side can
 * send a batch and then read all of its results.
 */

#include "fpu/softfloat.c"

/* A request with this op produces no result and flushes the results so far. */
#define OP_FLUSH 0xffff

enum {
    F16, BF16, F32, F64, X80, F128,
};

enum {
    OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_MULADD, OP_SQRT, OP_REM, OP_SCALBN,
    OP_MINMAX, OP_COMPARE, OP_COMPARE_QUIET, OP_ROUND_TO_INT, OP_LOG2,
    OP_TO_I8, OP_TO_I16, OP_TO_I32, OP_TO_I64,
    OP_TO_U8, OP_TO_U16, OP_TO_U32, OP_TO_U64,
    OP_FROM_I64, OP_FROM_U64,
    OP_TO_I128, OP_TO_U128, OP_FROM_I128, OP_FROM_U128,
    OP_TO_F16, OP_TO_BF16, OP_TO_F32, OP_TO_F64, OP_TO_X80, OP_TO_F128,
    OP_MODULO_I32, OP_MODULO_I64, OP_X80_MOD, OP_X80_ROUND,
    OP_X80_ROUND_AND_PACK, OP_X80_NORM_ROUND_AND_PACK,
    OP_PREDICATES, OP_DEFAULT_NAN, OP_SILENCE_NAN,
};

static uint64_t rd64(const unsigned char *p)
{
    uint64_t v;
    memcpy(&v, p, 8);
    return v;
}

static int32_t rd32(const unsigned char *p)
{
    int32_t v;
    memcpy(&v, p, 4);
    return v;
}

static uint16_t rd16(const unsigned char *p)
{
    uint16_t v;
    memcpy(&v, p, 2);
    return v;
}

static floatx80 x80(uint64_t lo, uint64_t hi)
{
    floatx80 r;
    r.low = lo;
    r.high = (uint16_t)hi;
    return r;
}

static void unsupported(unsigned op)
{
    fprintf(stderr, "driver: unsupported op %u (format %u, operation %u)\n",
            op, op / 64, op % 64);
    exit(2);
}

#define PRED(a, fmt)                                                    \
    ((uint64_t)fmt##_is_any_nan(a) |                                    \
     ((uint64_t)fmt##_is_quiet_nan(a, &s) << 1) |                       \
     ((uint64_t)fmt##_is_signaling_nan(a, &s) << 2) |                   \
     ((uint64_t)fmt##_is_zero(a) << 4) |                                \
     ((uint64_t)fmt##_is_neg(a) << 5) |                                 \
     ((uint64_t)fmt##_is_zero_or_denormal(a) << 6))

int main(void)
{
    unsigned char in[80];
    unsigned char out[32];

    for (;;) {
        size_t got = fread(in, 1, sizeof(in), stdin);
        if (got == 0) {
            break;
        }
        if (got != sizeof(in)) {
            fprintf(stderr, "driver: short record\n");
            return 1;
        }

        unsigned op = rd16(in);
        if (op == OP_FLUSH) {
            fflush(stdout);
            continue;
        }

        float_status s;
        memset(&s, 0, sizeof(s));
        s.float_rounding_mode = in[2];
        s.floatx80_rounding_precision = in[3];
        s.flush_to_zero = in[4];
        s.flush_inputs_to_zero = in[5];
        s.default_nan_mode = in[6];
        s.tininess_before_rounding = in[7];
        s.ftz_before_rounding = in[8];
        s.float_snan_rule = in[9];
        s.float_2nan_prop_rule = in[10];
        s.float_3nan_prop_rule = in[11];
        s.float_infzeronan_rule = in[12];
        s.floatx80_behaviour = in[13];
        s.default_nan_pattern = in[14];
        FloatRoundMode oprm = in[15];
        s.float_exception_flags = rd16(in + 16);
        s.rebias_overflow = in[18];
        s.rebias_underflow = in[19];
        int32_t imm = rd32(in + 20);
        int32_t imm2 = rd32(in + 24);
        uint64_t alo = rd64(in + 32), ahi = rd64(in + 40);
        uint64_t blo = rd64(in + 48), bhi = rd64(in + 56);
        uint64_t clo = rd64(in + 64), chi = rd64(in + 72);

        uint64_t rlo = 0, rhi = 0, extra = 0;
        unsigned fmt = op / 64, o = op % 64;

#define ARITH(T, fmt_, wrap, unwrap_lo, unwrap_hi)                          \
        {                                                               \
            T a = wrap(alo, ahi), b = wrap(blo, bhi);                   \
            T r;                                                        \
            (void)b;                                                    \
            switch (o) {                                                \
            case OP_ADD: r = fmt_##_add(a, b, &s); break;               \
            case OP_SUB: r = fmt_##_sub(a, b, &s); break;               \
            case OP_MUL: r = fmt_##_mul(a, b, &s); break;               \
            case OP_DIV: r = fmt_##_div(a, b, &s); break;               \
            case OP_SQRT: r = fmt_##_sqrt(a, &s); break;                \
            case OP_SCALBN: r = fmt_##_scalbn(a, imm, &s); break;       \
            case OP_ROUND_TO_INT: r = fmt_##_round_to_int(a, &s); break; \
            case OP_COMPARE:                                            \
                rlo = (uint64_t)(int64_t)fmt_##_compare(a, b, &s);      \
                goto done;                                              \
            case OP_COMPARE_QUIET:                                      \
                rlo = (uint64_t)(int64_t)fmt_##_compare_quiet(a, b, &s); \
                goto done;                                              \
            default: goto special;                                      \
            }                                                           \
            rlo = unwrap_lo(r);                                         \
            rhi = unwrap_hi(r);                                         \
            goto done;                                                  \
        }

#define W64(lo, hi) (lo)
#define WX80(lo, hi) x80(lo, hi)
#define W128(lo, hi) make_float128(hi, lo)
#define LO64(r) ((uint64_t)(r))
#define HI0(r) ((uint64_t)0)
#define LOX80(r) ((r).low)
#define HIX80(r) ((uint64_t)(r).high)
#define LO128(r) ((r).low)
#define HI128(r) ((r).high)

        switch (fmt) {
        case F16: ARITH(float16, float16, W64, LO64, HI0)
        case BF16: ARITH(bfloat16, bfloat16, W64, LO64, HI0)
        case F32: ARITH(float32, float32, W64, LO64, HI0)
        case F64: ARITH(float64, float64, W64, LO64, HI0)
        case X80: ARITH(floatx80, floatx80, WX80, LOX80, HIX80)
        case F128: ARITH(float128, float128, W128, LO128, HI128)
        default: unsupported(op);
        }

    special:
        switch (op) {
        /* Fused multiply add. */
        case F16 * 64 + OP_MULADD:
            rlo = imm2 ? float16_muladd_scalbn(alo, blo, clo, imm2, imm, &s)
                       : float16_muladd(alo, blo, clo, imm, &s);
            break;
        case BF16 * 64 + OP_MULADD:
            rlo = bfloat16_muladd(alo, blo, clo, imm, &s);
            break;
        case F32 * 64 + OP_MULADD:
            rlo = imm2 ? float32_muladd_scalbn(alo, blo, clo, imm2, imm, &s)
                       : float32_muladd(alo, blo, clo, imm, &s);
            break;
        case F64 * 64 + OP_MULADD:
            rlo = imm2 ? float64_muladd_scalbn(alo, blo, clo, imm2, imm, &s)
                       : float64_muladd(alo, blo, clo, imm, &s);
            break;
        case F128 * 64 + OP_MULADD: {
            float128 r = float128_muladd(make_float128(ahi, alo),
                                         make_float128(bhi, blo),
                                         make_float128(chi, clo), imm, &s);
            rlo = r.low;
            rhi = r.high;
            break;
        }

        /* Remainder. */
        case F32 * 64 + OP_REM: rlo = float32_rem(alo, blo, &s); break;
        case F64 * 64 + OP_REM: rlo = float64_rem(alo, blo, &s); break;
        case F128 * 64 + OP_REM: {
            float128 r = float128_rem(make_float128(ahi, alo),
                                      make_float128(bhi, blo), &s);
            rlo = r.low;
            rhi = r.high;
            break;
        }
        case X80 * 64 + OP_REM: {
            floatx80 r = floatx80_rem(x80(alo, ahi), x80(blo, bhi), &s);
            rlo = r.low;
            rhi = r.high;
            break;
        }
        case X80 * 64 + OP_X80_MOD: {
            uint64_t q;
            floatx80 r = floatx80_modrem(x80(alo, ahi), x80(blo, bhi),
                                         (imm & 1) == 0, &q, &s);
            rlo = r.low;
            rhi = r.high;
            extra = q;
            break;
        }

        /* Min and max. */
        case F16 * 64 + OP_MINMAX: rlo = float16_minmax(alo, blo, &s, imm); break;
        case BF16 * 64 + OP_MINMAX: rlo = bfloat16_minmax(alo, blo, &s, imm); break;
        case F32 * 64 + OP_MINMAX: rlo = float32_minmax(alo, blo, &s, imm); break;
        case F64 * 64 + OP_MINMAX: rlo = float64_minmax(alo, blo, &s, imm); break;
        case F128 * 64 + OP_MINMAX: {
            float128 r = float128_minmax(make_float128(ahi, alo),
                                         make_float128(bhi, blo), &s, imm);
            rlo = r.low;
            rhi = r.high;
            break;
        }

        case F32 * 64 + OP_LOG2: rlo = float32_log2(alo, &s); break;
        case F64 * 64 + OP_LOG2: rlo = float64_log2(alo, &s); break;

        /* Float to integer. */
#define TOI(F, name, T, fn) \
        case F * 64 + name: rlo = (uint64_t)(T)fn(alo, oprm, imm, &s); break;
        TOI(F16, OP_TO_I8, int64_t, float16_to_int8_scalbn)
        TOI(F16, OP_TO_I16, int64_t, float16_to_int16_scalbn)
        TOI(F16, OP_TO_I32, int64_t, float16_to_int32_scalbn)
        TOI(F16, OP_TO_I64, int64_t, float16_to_int64_scalbn)
        TOI(F16, OP_TO_U8, uint64_t, float16_to_uint8_scalbn)
        TOI(F16, OP_TO_U16, uint64_t, float16_to_uint16_scalbn)
        TOI(F16, OP_TO_U32, uint64_t, float16_to_uint32_scalbn)
        TOI(F16, OP_TO_U64, uint64_t, float16_to_uint64_scalbn)
        TOI(BF16, OP_TO_I8, int64_t, bfloat16_to_int8_scalbn)
        TOI(BF16, OP_TO_I16, int64_t, bfloat16_to_int16_scalbn)
        TOI(BF16, OP_TO_I32, int64_t, bfloat16_to_int32_scalbn)
        TOI(BF16, OP_TO_I64, int64_t, bfloat16_to_int64_scalbn)
        TOI(BF16, OP_TO_U8, uint64_t, bfloat16_to_uint8_scalbn)
        TOI(BF16, OP_TO_U16, uint64_t, bfloat16_to_uint16_scalbn)
        TOI(BF16, OP_TO_U32, uint64_t, bfloat16_to_uint32_scalbn)
        TOI(BF16, OP_TO_U64, uint64_t, bfloat16_to_uint64_scalbn)
        TOI(F32, OP_TO_I16, int64_t, float32_to_int16_scalbn)
        TOI(F32, OP_TO_I32, int64_t, float32_to_int32_scalbn)
        TOI(F32, OP_TO_I64, int64_t, float32_to_int64_scalbn)
        TOI(F32, OP_TO_U16, uint64_t, float32_to_uint16_scalbn)
        TOI(F32, OP_TO_U32, uint64_t, float32_to_uint32_scalbn)
        TOI(F32, OP_TO_U64, uint64_t, float32_to_uint64_scalbn)
        TOI(F64, OP_TO_I16, int64_t, float64_to_int16_scalbn)
        TOI(F64, OP_TO_I32, int64_t, float64_to_int32_scalbn)
        TOI(F64, OP_TO_I64, int64_t, float64_to_int64_scalbn)
        TOI(F64, OP_TO_U16, uint64_t, float64_to_uint16_scalbn)
        TOI(F64, OP_TO_U32, uint64_t, float64_to_uint32_scalbn)
        TOI(F64, OP_TO_U64, uint64_t, float64_to_uint64_scalbn)

        case F128 * 64 + OP_TO_I32:
            rlo = (uint64_t)(int64_t)float128_to_int32_scalbn(
                make_float128(ahi, alo), oprm, imm, &s);
            break;
        case F128 * 64 + OP_TO_I64:
            rlo = (uint64_t)float128_to_int64_scalbn(make_float128(ahi, alo),
                                                     oprm, imm, &s);
            break;
        case F128 * 64 + OP_TO_U32:
            rlo = float128_to_uint32_scalbn(make_float128(ahi, alo), oprm, imm,
                                            &s);
            break;
        case F128 * 64 + OP_TO_U64:
            rlo = float128_to_uint64_scalbn(make_float128(ahi, alo), oprm, imm,
                                            &s);
            break;
        case F128 * 64 + OP_TO_I128: {
            Int128 r = float128_to_int128_scalbn(make_float128(ahi, alo), oprm,
                                                 imm, &s);
            rlo = int128_getlo(r);
            rhi = int128_gethi(r);
            break;
        }
        case F128 * 64 + OP_TO_U128: {
            Int128 r = float128_to_uint128_scalbn(make_float128(ahi, alo), oprm,
                                                  imm, &s);
            rlo = int128_getlo(r);
            rhi = int128_gethi(r);
            break;
        }
        case X80 * 64 + OP_TO_I32:
            rlo = (uint64_t)(int64_t)floatx80_to_int32_scalbn(x80(alo, ahi),
                                                              oprm, imm, &s);
            break;
        case X80 * 64 + OP_TO_I64:
            rlo = (uint64_t)floatx80_to_int64_scalbn(x80(alo, ahi), oprm, imm,
                                                     &s);
            break;
        case F64 * 64 + OP_MODULO_I32:
            rlo = (uint64_t)(int64_t)float64_to_int32_modulo(alo, oprm, &s);
            break;
        case F64 * 64 + OP_MODULO_I64:
            rlo = (uint64_t)float64_to_int64_modulo(alo, oprm, &s);
            break;

        /* Integer to float. */
        case F16 * 64 + OP_FROM_I64:
            rlo = int64_to_float16_scalbn(alo, imm, &s);
            break;
        case F16 * 64 + OP_FROM_U64:
            rlo = uint64_to_float16_scalbn(alo, imm, &s);
            break;
        case BF16 * 64 + OP_FROM_I64:
            rlo = int64_to_bfloat16_scalbn(alo, imm, &s);
            break;
        case BF16 * 64 + OP_FROM_U64:
            rlo = uint64_to_bfloat16_scalbn(alo, imm, &s);
            break;
        case F32 * 64 + OP_FROM_I64:
            rlo = int64_to_float32_scalbn(alo, imm, &s);
            break;
        case F32 * 64 + OP_FROM_U64:
            rlo = uint64_to_float32_scalbn(alo, imm, &s);
            break;
        case F64 * 64 + OP_FROM_I64:
            rlo = int64_to_float64_scalbn(alo, imm, &s);
            break;
        case F64 * 64 + OP_FROM_U64:
            rlo = uint64_to_float64_scalbn(alo, imm, &s);
            break;
        case F128 * 64 + OP_FROM_I64: {
            float128 r = int64_to_float128(alo, &s);
            rlo = r.low;
            rhi = r.high;
            break;
        }
        case F128 * 64 + OP_FROM_U64: {
            float128 r = uint64_to_float128(alo, &s);
            rlo = r.low;
            rhi = r.high;
            break;
        }
        case F128 * 64 + OP_FROM_I128: {
            float128 r = int128_to_float128(int128_make128(alo, ahi), &s);
            rlo = r.low;
            rhi = r.high;
            break;
        }
        case F128 * 64 + OP_FROM_U128: {
            float128 r = uint128_to_float128(int128_make128(alo, ahi), &s);
            rlo = r.low;
            rhi = r.high;
            break;
        }
        case X80 * 64 + OP_FROM_I64: {
            floatx80 r = int64_to_floatx80(alo, &s);
            rlo = r.low;
            rhi = r.high;
            break;
        }

        /* Float to float. */
        case F16 * 64 + OP_TO_F32: rlo = float16_to_float32(alo, imm, &s); break;
        case F16 * 64 + OP_TO_F64: rlo = float16_to_float64(alo, imm, &s); break;
        case BF16 * 64 + OP_TO_F32: rlo = bfloat16_to_float32(alo, &s); break;
        case BF16 * 64 + OP_TO_F64: rlo = bfloat16_to_float64(alo, &s); break;
        case F32 * 64 + OP_TO_F16: rlo = float32_to_float16(alo, imm, &s); break;
        case F32 * 64 + OP_TO_BF16: rlo = float32_to_bfloat16(alo, &s); break;
        case F32 * 64 + OP_TO_F64: rlo = float32_to_float64(alo, &s); break;
        case F64 * 64 + OP_TO_F16: rlo = float64_to_float16(alo, imm, &s); break;
        case F64 * 64 + OP_TO_BF16: rlo = float64_to_bfloat16(alo, &s); break;
        case F64 * 64 + OP_TO_F32: rlo = float64_to_float32(alo, &s); break;
        case F32 * 64 + OP_TO_X80: {
            floatx80 r = float32_to_floatx80(alo, &s);
            rlo = r.low;
            rhi = r.high;
            break;
        }
        case F64 * 64 + OP_TO_X80: {
            floatx80 r = float64_to_floatx80(alo, &s);
            rlo = r.low;
            rhi = r.high;
            break;
        }
        case F32 * 64 + OP_TO_F128: {
            float128 r = float32_to_float128(alo, &s);
            rlo = r.low;
            rhi = r.high;
            break;
        }
        case F64 * 64 + OP_TO_F128: {
            float128 r = float64_to_float128(alo, &s);
            rlo = r.low;
            rhi = r.high;
            break;
        }
        case X80 * 64 + OP_TO_F32: rlo = floatx80_to_float32(x80(alo, ahi), &s); break;
        case X80 * 64 + OP_TO_F64: rlo = floatx80_to_float64(x80(alo, ahi), &s); break;
        case X80 * 64 + OP_TO_F128: {
            float128 r = floatx80_to_float128(x80(alo, ahi), &s);
            rlo = r.low;
            rhi = r.high;
            break;
        }
        case F128 * 64 + OP_TO_F32:
            rlo = float128_to_float32(make_float128(ahi, alo), &s);
            break;
        case F128 * 64 + OP_TO_F64:
            rlo = float128_to_float64(make_float128(ahi, alo), &s);
            break;
        case F128 * 64 + OP_TO_X80: {
            floatx80 r = float128_to_floatx80(make_float128(ahi, alo), &s);
            rlo = r.low;
            rhi = r.high;
            break;
        }

        /* floatx80 specials. */
        case X80 * 64 + OP_X80_ROUND: {
            floatx80 r = floatx80_round(x80(alo, ahi), &s);
            rlo = r.low;
            rhi = r.high;
            break;
        }
        case X80 * 64 + OP_X80_ROUND_AND_PACK: {
            floatx80 r = roundAndPackFloatx80(s.floatx80_rounding_precision,
                                              imm2 & 1, imm, alo, ahi, &s);
            rlo = r.low;
            rhi = r.high;
            break;
        }
        case X80 * 64 + OP_X80_NORM_ROUND_AND_PACK: {
            floatx80 r = normalizeRoundAndPackFloatx80(
                s.floatx80_rounding_precision, imm2 & 1, imm, alo, ahi, &s);
            rlo = r.low;
            rhi = r.high;
            break;
        }

        /* Predicates and NaN helpers. */
        case F16 * 64 + OP_PREDICATES:
            rlo = PRED((float16)alo, float16) |
                  ((uint64_t)float16_is_infinity(alo) << 3) |
                  ((uint64_t)float16_is_normal(alo) << 7);
            break;
        case BF16 * 64 + OP_PREDICATES:
            rlo = PRED((bfloat16)alo, bfloat16) |
                  ((uint64_t)bfloat16_is_infinity(alo) << 3) |
                  ((uint64_t)bfloat16_is_normal(alo) << 7);
            break;
        case F32 * 64 + OP_PREDICATES:
            rlo = PRED((float32)alo, float32) |
                  ((uint64_t)float32_is_infinity(alo) << 3) |
                  ((uint64_t)float32_is_normal(alo) << 7) |
                  ((uint64_t)float32_is_denormal(alo) << 8);
            break;
        case F64 * 64 + OP_PREDICATES:
            rlo = PRED((float64)alo, float64) |
                  ((uint64_t)float64_is_infinity(alo) << 3) |
                  ((uint64_t)float64_is_normal(alo) << 7) |
                  ((uint64_t)float64_is_denormal(alo) << 8);
            break;
        case F128 * 64 + OP_PREDICATES: {
            float128 a = make_float128(ahi, alo);
            rlo = PRED(a, float128) |
                  ((uint64_t)float128_is_infinity(a) << 3) |
                  ((uint64_t)float128_is_normal(a) << 7) |
                  ((uint64_t)float128_is_denormal(a) << 8);
            break;
        }
        case X80 * 64 + OP_PREDICATES: {
            floatx80 a = x80(alo, ahi);
            rlo = PRED(a, floatx80) |
                  ((uint64_t)floatx80_is_infinity(a, &s) << 3) |
                  ((uint64_t)floatx80_invalid_encoding(a, &s) << 9);
            break;
        }
        case F16 * 64 + OP_DEFAULT_NAN: rlo = float16_default_nan(&s); break;
        case BF16 * 64 + OP_DEFAULT_NAN: rlo = bfloat16_default_nan(&s); break;
        case F32 * 64 + OP_DEFAULT_NAN: rlo = float32_default_nan(&s); break;
        case F64 * 64 + OP_DEFAULT_NAN: rlo = float64_default_nan(&s); break;
        case F128 * 64 + OP_DEFAULT_NAN: {
            float128 r = float128_default_nan(&s);
            rlo = r.low;
            rhi = r.high;
            break;
        }
        case X80 * 64 + OP_DEFAULT_NAN: {
            floatx80 r = floatx80_default_nan(&s);
            rlo = r.low;
            rhi = r.high;
            break;
        }
        case F16 * 64 + OP_SILENCE_NAN: rlo = float16_silence_nan(alo, &s); break;
        case BF16 * 64 + OP_SILENCE_NAN: rlo = bfloat16_silence_nan(alo, &s); break;
        case F32 * 64 + OP_SILENCE_NAN: rlo = float32_silence_nan(alo, &s); break;
        case F64 * 64 + OP_SILENCE_NAN: rlo = float64_silence_nan(alo, &s); break;
        case F128 * 64 + OP_SILENCE_NAN: {
            float128 r = float128_silence_nan(make_float128(ahi, alo), &s);
            rlo = r.low;
            rhi = r.high;
            break;
        }
        case X80 * 64 + OP_SILENCE_NAN: {
            floatx80 r = floatx80_silence_nan(x80(alo, ahi), &s);
            rlo = r.low;
            rhi = r.high;
            break;
        }

        default:
            unsupported(op);
        }

    done:
        memset(out, 0, sizeof(out));
        memcpy(out, &rlo, 8);
        memcpy(out + 8, &rhi, 8);
        memcpy(out + 16, &extra, 8);
        {
            uint16_t f = s.float_exception_flags;
            memcpy(out + 24, &f, 2);
        }
        if (fwrite(out, 1, sizeof(out), stdout) != sizeof(out)) {
            return 1;
        }
    }
    return 0;
}
