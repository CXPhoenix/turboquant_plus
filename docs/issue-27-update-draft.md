# Issue #27 Update Draft (do not post yet)

## Proposed new title

**TurboQuant: missing q8_0 × turbo FA kernel instantiations + quality guidance for Q4_K_M models**

## Comment draft

### Resolution

Two issues identified and addressed:

**1. Missing q8_0 × turbo Metal FA kernel instantiations (bug, fixed)**

The asymmetric K/V implementation only added turbo × turbo kernel pairs. q8_0 × turbo pairs were missing, causing silent dispatch failures when using e.g. `-ctk q8_0 -ctv turbo4`. turbo4-V produced NaN; turbo3-V fell to an accidental fallback that happened to work.

Fix: added 150 new Metal FA kernel instantiations for all q8_0 × turbo combinations (both directions, all head dims, vec and non-vec paths). Updated gatekeeper and assertion to allow these pairs.

**2. Quantization stacking on Q4_K_M models (quality issue, mitigated via asymmetric K/V)**

Original M2/Apple8 hardware bug hypothesis retired. Same-model cross-hardware tests confirmed no M2-specific divergence. The quality collapse reproduces identically on M5 and M2.

Root cause: Q4_K_M weight quantization + turbo K quantization stacks too aggressively. K determines attention routing via softmax(Q·K^T), so K errors cascade exponentially. V errors are proportional and tolerable.

**Rescue: q8_0-K + turbo-V**

Qwen2.5-7B-Instruct-Q4_K_M results:

| K | V | PPL | vs q8_0 (6.58) | V compression |
|---|---|------|----------------|---------------|
| q8_0 | q8_0 | 6.58 | baseline | 1.0x |
| q8_0 | turbo4 | 6.64 | +1.0% | 2.0x |
| q8_0 | turbo3 | 6.71 | +2.0% | 2.3x |
| q8_0 | turbo2 | 6.91 | +5.1% | 3.2x |
| turbo4 | turbo4 | 218 | catastrophic | — |
| turbo3 | turbo3 | 3556 | catastrophic | — |

Users with Q4_K_M models should use `-ctk q8_0 -ctv turbo3` (or turbo4/turbo2) for V compression with near-baseline quality. Symmetric turbo remains correct for Q8_0+ weight models.

### Files changed

- `ggml/src/ggml-metal/ggml-metal.metal` — 150 new q8_0 × turbo kernel instantiations
- `ggml/src/ggml-metal/ggml-metal-device.m` — gatekeeper allows q8_0 × turbo pairs
- `ggml/src/ggml-metal/ggml-metal-device.cpp` — pipeline naming (already handled by k/v prefix format)
- `ggml/src/ggml-metal/ggml-metal-ops.cpp` — assertion allows q8_0 × turbo pairs
