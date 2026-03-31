# TurboQuant VX: 1-Bit Sign-Only V Cache Experiment

## Summary

We implemented and tested a 1-bit sign-only V cache format (TURBO_VX) that stores only the sign of each WHT-rotated value element. At 1.5 bits/val (10.67x compression vs fp16, 5.7x vs q8_0), it is the most aggressive V compression we've tested.

**Result:** 1-bit V is at the boundary of viability. Usable on strong base-weight models (+20% PPL) but too aggressive for sensitive low-bit models (+52% PPL). turbo2-V (2-bit, +3-5% PPL) remains the practical compression floor.

## Format

```c
typedef struct {
    ggml_fp16_t norm;           // 2 bytes: corrected group norm
    uint8_t     signs[4];       // 4 bytes: 32 sign bits
} block_turbo_vx_0;             // 6 bytes per 32 elements = 1.5 bits/val
```

**Dequant:** `value = sign * norm * EXPECTED_MAGNITUDE`

where `EXPECTED_MAGNITUDE = sqrt(2/pi) / sqrt(128) ≈ 0.070523` is the expected absolute value of a unit-Gaussian WHT-rotated element.

**Norm correction:** `corrected_norm = group_L2_norm / sqrt(2/pi)` because the reconstruction assumes all elements have the expected magnitude.

## Rationale

WHT rotation Gaussianizes the KV vectors (kurtosis 900 → 2.9, validated on real Qwen3 tensors). For a Gaussian, the sign captures ~80% of element variance (`E[x * sign(x)] / E[x^2] = sqrt(2/pi)`). The remaining ~20% is magnitude variation.

Previous empirical work showed V errors are proportional through the attention weighted sum, not exponential through softmax like K errors. This suggests V can tolerate the 20% information loss from discarding magnitudes.

## Results

All tests on M5 Max. Wikitext-2, 512 context. q8_0-K in all cases.

### phi-4-14B (Q8_0 weights)

| V type | PPL | vs q8_0/q8_0 (4.69) | V memory (512 ctx) | V compression |
|--------|------|---------------------|--------------------|--------------|
| q8_0 | 4.69 | baseline | 106.25 MiB | 1.0x |
| turbo4 | 4.70 | +0.3% | ~53 MiB | 2.0x |
| turbo3 | 4.74 | +1.1% | ~46 MiB | 2.3x |
| turbo2 | 4.84 | +3.1% | ~40 MiB | 2.7x |
| **turbo_vx** | **5.61** | **+19.6%** | **18.75 MiB** | **5.7x** |

### Qwen2.5-7B-Instruct (Q4_K_M weights — sensitive model)

| V type | PPL | vs q8_0/q8_0 (6.58) | V memory (512 ctx) | V compression |
|--------|------|---------------------|--------------------|--------------|
| q8_0 | 6.58 | baseline | 29.75 MiB | 1.0x |
| turbo4 | 6.64 | +1.0% | ~15 MiB | 2.0x |
| turbo3 | 6.71 | +2.0% | ~13 MiB | 2.3x |
| turbo2 | 6.91 | +5.1% | ~11 MiB | 2.7x |
| **turbo_vx** | **10.00** | **+52%** | **5.25 MiB** | **5.7x** |

## Analysis

### What the sign captures

In WHT space, the sign carries the direction of each element's contribution to attention. For V values weighted by softmax, direction matters more than magnitude because:
- Positive/negative contributions largely cancel in the weighted sum
- The attention-weight distribution is typically sparse (few tokens dominate)
- The dominant tokens' V signs contribute the most to the output

### Why +20% PPL on phi-4 but +52% on Qwen Q4_K_M

The quality difference tracks the existing sensitivity pattern. Qwen2.5-7B Q4_K_M is already sensitive to V compression (turbo2-V gives +5.1% PPL). Reducing from 2-bit to 1-bit loses the centroid magnitude information that turbo2 preserves. On this model, the magnitude variation in WHT-rotated V elements carries meaningful information that sign-only discards.

phi-4 with Q8_0 weights produces cleaner V activations. The WHT-rotated elements are more uniformly distributed around the expected magnitude, so the sign-only approximation loses less.

### The compression-quality curve

| V bits/val | Format | phi-4 PPL | Qwen7B PPL | V compression |
|-----------|--------|-----------|------------|---------------|
| 8.5 | q8_0 | 4.69 | 6.58 | 1.0x |
| 4.25 | turbo4 | 4.70 | 6.64 | 2.0x |
| 3.5 | turbo3 | 4.74 | 6.71 | 2.3x |
| 2.5 | turbo2 | 4.84 | 6.91 | 2.7x |
| **1.5** | **turbo_vx** | **5.61** | **10.00** | **5.7x** |

The curve shows diminishing returns below 2.5 bits/val. The jump from turbo2 (2.5) to turbo_vx (1.5) costs more quality per bit than any previous step. This suggests 2-bit is near the information-theoretic floor for V compression via PolarQuant + WHT.

## Implementation Notes

- Block format: 6 bytes per 32 elements (2B norm + 4B signs)
- Dequant: 3 operations per element (read sign bit, multiply by constant, apply sign). No LUT, no centroid lookup.
- Metal FA dequant is simpler and faster than any existing turbo type
- SET_ROWS kernel: same normalize → WHT → extract pattern as turbo2/3/4, but only stores signs instead of centroid indices
- 25 new FA kernel instantiations (15 non-vec + 10 vec) for `kq8_0_vturbo_vx` pairs
- Total implementation: ~200 lines across 10 files

## Conclusion

1-bit sign-only V is the compression floor. It works (PPL is finite, not NaN) but the quality cost is too high for general use. **turbo2-V remains the practical V compression limit.**

The experiment is informative for three reasons:
1. It establishes that V can survive extreme compression — 1-bit is not catastrophic
2. It confirms the diminishing-returns curve below 2 bits
3. The format's extreme simplicity (no LUT, no centroids) could serve as a fast-path for sparse V skip evaluation in future work

## Files Modified

- `ggml/include/ggml.h` — type enum
- `ggml/src/ggml-common.h` — block format
- `ggml/src/ggml.c` — type traits + quantize dispatch
- `ggml/src/ggml-quants.h` — function declarations
- `ggml/src/ggml-turbo-quant.c` — CPU quantize/dequant
- `ggml/src/ggml-metal/ggml-metal.metal` — FA dequant + SET_ROWS kernel + 25 FA instantiations
- `ggml/src/ggml-metal/ggml-metal-device.m` — supports_op
- `ggml/src/ggml-metal/ggml-metal-ops.cpp` — assertion
- `src/llama-graph.cpp` — turbo type checks
- `src/llama-kv-cache.cpp` — turbo type checks
- `src/llama-context.cpp` — turbo type checks
- `common/arg.cpp` — cache type parsing
- `tools/llama-bench/llama-bench.cpp` — type parsing
