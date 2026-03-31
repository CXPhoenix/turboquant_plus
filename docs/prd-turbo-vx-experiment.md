# PRD: V-Only Aggressive KV Cache Format (TURBO_VX)

## 1. Problem

Current empirical evidence from this repo shows a stark asymmetry in how K and V tolerate quantization:

```
Qwen2.5-7B Q4_K_M:
  turbo3-K + q8_0-V:  PPL 3556    (K errors cascade through softmax)
  q8_0-K + turbo3-V:  PPL 6.71    (+2.0% — V errors are proportional)
  q8_0-K + turbo2-V:  PPL 6.91    (+5.1% — still usable at 2-bit)
```

K controls attention routing via `softmax(Q * K^T)`. Small K errors shift which tokens dominate, cascading exponentially. V errors scale linearly through the weighted sum. This means V has significantly more compression headroom than we're currently using.

The most aggressive V option today is `q8_0-K + turbo2-V` at 2.5 bits/val (+5.1% PPL). The question: can we go below 2 bits on V while keeping quality bounded?

This should be a V-only experiment because:
- K compression is already at its practical limit for sensitive models
- The asymmetric infrastructure already exists and is validated
- V-only means we can iterate without touching any K codepath

## 2. Goals

- Create a local experimental V-only cache type more aggressive than turbo2
- Target ~1.5 bits/val or lower (vs turbo2's 2.5 bits/val)
- Pair with `q8_0-K` for all validation (proven safe K path)
- Integrate with current Metal FA vec kernel path
- Measure quality/compression tradeoff against existing `q8_0/turbo2`, `q8_0/turbo3`, `q8_0/turbo4`
- Avoid any changes to existing turbo2/3/4 codepaths

## 3. Non-goals

- No full upstream productization
- No CUDA/Vulkan implementation in v1
- No refactor of the turbo type family
- No changes to public recommendations until data exists
- No interference with sparse V (orthogonal optimization)
- No turbo5/turbo6 public naming — internal experimental name only

## 4. Design Constraints from Current Codebase

**128-element WHT groups:** All turbo types operate on 128-element groups with WHT rotation. VX must use the same rotation to remain compatible with the existing Q pre-rotation in `llama-graph.cpp` and the inverse WHT in `build_attn_mha`. The SET_ROWS kernel already handles 128-element groups for all turbo types.

**32-element sub-blocks:** All existing turbo block types use QK=32 (32 elements per block, 4 blocks per 128-element WHT group). The Metal FA vec kernel's dequant loop is structured around this. VX should use the same QK=32 to fit the existing template without special-casing.

**FA kernel architecture:** The FA vec kernel calls a dequant function per block (`dequantize_*_t4`), producing a `type4` (4 floats). The dequant function receives a block pointer and an element index. VX needs to provide this interface. The kernel doesn't care about the internal block format.

**Mixed K/V support:** The `k{type}_v{type}` pipeline naming and kernel instantiation system already handles arbitrary K/V type combinations. Adding VX means adding new `kq8_0_vturbo_vx` instantiations in the same pattern.

**InnerQ tensor:** The `innerq_scale_inv` tensor exists in the graph for per-channel equalization. VX should pass it through unchanged (or ignore it — InnerQ is optional).

**Metal hot path sensitivity:** The FA dequant is the inner loop. VX dequant must be at least as fast as turbo2 dequant, ideally faster (simpler format = fewer operations).

**Regression risk:** phi-4-Q8_0 and Qwen2.5-7B Q4_K_M are the primary regression gates. No existing symmetric or asymmetric config should change behavior.

## 5. Candidate Format Options

### Candidate A: 1-bit Sign-Only (RECOMMENDED)

**Concept:** In WHT space, elements are approximately N(0, 1/sqrt(128)). The sign carries the most information. Store only the sign bit per element plus a corrected group norm.

**Block format:**
```c
typedef struct {
    ggml_fp16_t norm;        // 2 bytes: corrected group norm
    uint8_t     signs[4];    // 4 bytes: 32 sign bits (1 per element)
    // total: 6 bytes per 32 elements
} block_turbo_vx_0;
```

**Dequant:** `value = (sign ? +1 : -1) * norm * EXPECTED_MAGNITUDE`, where `EXPECTED_MAGNITUDE = E[|x|]` for N(0, 1/sqrt(128)) ≈ 0.0705 (= sqrt(2/pi) / sqrt(128)).

This is a single constant multiply per element. No LUT, no centroid lookup, no index extraction. The fastest possible turbo dequant.

**Bits/val:** 6 * 8 / 32 = **1.5 bits/val**
**Compression vs fp16:** 10.67x
**Compression vs q8_0:** 5.67x
**vs turbo2 (2.5 bits/val):** 40% less memory

**Implementation difficulty:** Low. Simpler than any existing turbo type. No qs byte array, no centroid tables. The dequant is trivial. SET_ROWS quantize is: normalize, WHT rotate, extract signs, compute corrected norm.

**Quality risk:** Medium-high. 1-bit is aggressive. WHT Gaussianization means the sign captures ~70% of the element variance (for a Gaussian, E[x * sign(x)] / E[x^2] = sqrt(2/pi) ≈ 0.80). The remaining 20-30% is magnitude variation. On V this might be acceptable; on K it would be catastrophic.

**Metal FA fit:** Perfect. The dequant function is 3 lines: read norm, read sign byte, multiply. No constant memory pressure, no LUT. Fastest possible decode.

### Candidate B: 1.5-bit Ternary (sign + zero)

**Concept:** Add a "zero" level: each element is +magnitude, -magnitude, or zero. 3 levels encoded in ~1.58 bits. Pack as 2 trits per 3 bits or use a 32-element ternary code.

**Block format:** Complex packing (e.g., 5 trits per byte, or 32 trits in 6 bytes). Awkward.

**Bits/val:** ~1.6-1.8 depending on packing
**Implementation difficulty:** Medium-high. Ternary packing is fiddly.
**Quality risk:** Lower than Candidate A (zero level helps with near-zero elements).
**Metal FA fit:** OK but dequant is more complex than Candidate A.

**Verdict:** More complexity for marginal quality gain. Skip for v1.

### Candidate C: 2-bit with shared norm (turbo2 with larger blocks)

**Concept:** Same 2-bit centroids as turbo2 but with 64 or 128 elements per block instead of 32, reducing norm overhead.

**Bits/val:** ~2.1-2.25 (vs turbo2's 2.5)
**Implementation difficulty:** Medium. Breaks the QK=32 assumption in FA templates.
**Quality risk:** Low (same quantization, less norm precision per sub-block).
**Metal FA fit:** Would require template changes for different block sizes.

**Verdict:** Modest memory win, non-trivial plumbing changes. Not worth the complexity for v1.

### Recommendation: Candidate A (1-bit Sign-Only)

Simplest format, largest compression jump, cleanest Metal integration. The quality risk is the main question mark — and that's exactly what the experiment is designed to answer.

## 6. Recommended v1 Scope

- Register `GGML_TYPE_TURBO_VX_0` as a new type
- Define `block_turbo_vx_0` (6 bytes per 32 elements)
- Implement CPU quantize reference (`quantize_row_turbo_vx_0_ref`)
- Implement CPU dequant reference (`dequantize_row_turbo_vx_0`)
- Implement Metal SET_ROWS kernel (`kernel_set_rows_turbo_vx`)
- Implement Metal FA vec dequant (`dequantize_turbo_vx_0_t4`)
- Add FA kernel instantiations for `kq8_0_vturbo_vx` (vec + non-vec, all head dims)
- Wire cache type parsing ("turbo_vx" or similar CLI name)
- Wire `supports_op` gatekeeper for the new type
- Run validation suite

**Not in v1:**
- No symmetric `turbo_vx/turbo_vx` (K should never use this format)
- No CUDA/Vulkan
- No MUL_MAT/MUL_MV kernels (not needed for KV cache path)
- No turbo_vx × turbo combinations (only q8_0 × turbo_vx)

## 7. File-by-File Implementation Plan

### `ggml/include/ggml.h`
- Add `GGML_TYPE_TURBO_VX_0` to the type enum

### `ggml/src/ggml-common.h`
- Define `QK_TURBO_VX` (32)
- Define `block_turbo_vx_0` struct: `{ ggml_half norm; uint8_t signs[4]; }`
- Add static_assert on block size (6 bytes)

### `ggml/src/ggml.c`
- Add type traits entry for `GGML_TYPE_TURBO_VX_0`: type_name = "turbo_vx", type_size = 6, blck_size = 32, to_float, from_float_ref

### `ggml/src/ggml-turbo-quant.c`
- `quantize_row_turbo_vx_0_ref`: normalize group, WHT rotate, extract signs, compute corrected norm (`grp_norm * sqrt(2/pi)` since reconstruction assumes all elements have the expected magnitude)
- `dequantize_row_turbo_vx_0`: for each element, `sign * norm * EXPECTED_MAG`
- `quantize_turbo_vx_0`: row-level wrapper

### `ggml/src/ggml-metal/ggml-metal.metal`
- `dequantize_turbo_vx_0_t4`: the FA vec dequant function
  ```metal
  constant float TURBO_VX_MAG = 0.070523f; // sqrt(2/pi) / sqrt(128)
  // ...
  float norm = float(xb->norm);
  uint8_t sb = xb->signs[il];
  uint8_t s0 = (sb >> (sshift + 0)) & 1;
  // ... extract 4 sign bits
  float mag = norm * TURBO_VX_MAG;
  reg = type4(float4(s0 ? mag : -mag, s1 ? mag : -mag, s2 ? mag : -mag, s3 ? mag : -mag));
  ```
- `kernel_set_rows_turbo_vx`: quantize kernel — normalize, WHT rotate, extract signs, store corrected norm
- FA kernel instantiations: `kq8_0_vturbo_vx_dk{N}_dv{N}` for all supported head dims (vec + non-vec)

### `ggml/src/ggml-metal/ggml-metal-device.m`
- Add `GGML_TYPE_TURBO_VX_0` to `supports_op` for SET_ROWS and FLASH_ATTN_EXT (when paired with supported K types)

### `ggml/src/ggml-metal/ggml-metal-device.cpp`
- Pipeline name construction already handles arbitrary types via `ggml_type_name` — should work automatically if type_name is "turbo_vx"

### `ggml/src/ggml-metal/ggml-metal-ops.cpp`
- Add `GGML_TYPE_TURBO_VX_0` to the asymmetric K/V assertion allowlist

### `src/llama-kv-cache.cpp`
- No changes needed — VX uses the same 128-element WHT group structure. KV cache allocation handles arbitrary types via `ggml_type_size`.

### `src/llama-context.cpp`
- Add "turbo_vx" to cache type validation/parsing

### `src/llama-graph.cpp`
- No changes needed — the V inverse WHT path already triggers on any turbo V type. Add `GGML_TYPE_TURBO_VX_0` to the turbo type checks.

### `common/arg.cpp`
- Add "turbo_vx" to cache type argument parsing

### `tools/llama-bench/llama-bench.cpp`
- Add "turbo_vx" to `ggml_type_from_name`

## 8. Validation Plan

All tests on M5 Max. Wikitext-2, 512 context, 4 chunks.

### Regression (must match existing baselines exactly)

| Model | K | V | Expected PPL |
|-------|---|---|-------------|
| phi-4-Q8_0 | q8_0 | q8_0 | 4.690 |
| phi-4-Q8_0 | turbo3 | turbo3 | 4.886 |
| phi-4-Q8_0 | q8_0 | turbo4 | 4.702 |
| Qwen2.5-7B Q4_K_M | q8_0 | q8_0 | 6.577 |
| Qwen2.5-7B Q4_K_M | q8_0 | turbo4 | 6.644 |

### Experimental (new format)

| Model | K | V | Comparison targets |
|-------|---|---|-------------------|
| phi-4-Q8_0 | q8_0 | turbo_vx | vs q8_0/turbo2 (4.835), q8_0/turbo3 (4.742) |
| Qwen2.5-7B Q4_K_M | q8_0 | turbo_vx | vs q8_0/turbo2 (6.911), q8_0/turbo3 (6.707) |

### Speed (one check)

| Model | K | V | Check |
|-------|---|---|-------|
| phi-4-Q8_0 | q8_0 | turbo_vx | prefill + decode, compare to q8_0/turbo3 |

VX dequant should be faster than turbo3 (fewer operations). If it's slower, something is wrong.

### Optional M2

One run of phi-4-Q8_0 q8_0/turbo_vx to confirm cross-hardware parity.

## 9. Success Criteria

**Gate 1 — No regression:** All existing configs in the regression table match baselines exactly.

**Gate 2 — Meaningful compression:** turbo_vx at 1.5 bits/val is a 40% memory reduction over turbo2 (2.5 bits/val) on V cache. This is the structural win.

**Gate 3 — Bounded quality loss:**
- On phi-4-Q8_0: PPL < 5.5 (i.e., < +17% vs q8_0 baseline). If worse than turbo2 (+3.1%), note but don't necessarily kill.
- On Qwen2.5-7B Q4_K_M: PPL < 8.0 (i.e., < +22% vs q8_0 baseline). If worse than turbo2 (+5.1%), note but don't necessarily kill.
- If PPL is catastrophic (>50 on either model), the format is not viable for V and the experiment is conclusive negative.

**Gate 4 — Speed neutral or positive:** Decode speed should not regress vs q8_0/turbo3. VX dequant is simpler so it should be at least as fast.

**Decision matrix:**
- All gates pass: VX is a viable V-only aggressive format. Proceed to documentation and optional productization.
- Gates 1-2 pass, Gate 3 marginal (worse than turbo2 but not catastrophic): VX has a niche for extreme memory pressure. Document as experimental.
- Gate 3 fails catastrophically: 1-bit sign-only is too aggressive for V. The experiment is informative (lower bound established) but the format is not worth shipping. Consider Candidate B (ternary) as follow-up.
- Gate 1 fails: implementation bug. Fix before evaluating.

## 10. Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| 1-bit is too aggressive for V, PPL catastrophic | Medium | Low (experiment is conclusive) | This is the whole point of the experiment. A negative result is still informative. |
| Hot-path Metal FA complexity | Low | Medium | VX dequant is simpler than turbo2. Fewer operations = lower risk. |
| Naming confusion with turbo family | Low | Low | Using `turbo_vx` (experimental) not `turbo5`. |
| Overfitting to phi-4 and Qwen2.5-7B | Medium | Medium | These are the same models used for all turbo validation. Consistent comparison. |
| VX is strictly worse than existing q8_0/turbo2 | Medium | Low | If so, we've established the compression floor. turbo2-V is the practical limit and VX is the evidence. |
| Adding type registration bloat | Low | Low | One new type, minimal. Can be removed if experiment is negative. |

## 11. Recommendation

**Implement prototype now.**

Rationale:
- The format is trivially simple (simpler than any existing turbo type)
- The Metal dequant is ~3 lines of code
- The asymmetric infrastructure already exists
- The validation models and baselines are already established
- Implementation time is estimated at 2-4 hours
- A negative result is nearly as valuable as a positive one (establishes the 1-bit V floor)

The main risk (quality) is exactly what the experiment measures. The implementation risk is minimal because the format is simpler than turbo2, and the codebase already handles arbitrary mixed K/V types.

If quality is catastrophic, we've learned that V needs at least ~2 bits (turbo2 is the floor). If quality is acceptable, we've found a 40% memory win on V cache for free. Either outcome justifies the 2-4 hours.
