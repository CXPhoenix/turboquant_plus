# TurboQuant Configuration Recommendations

Practical guidance for choosing TurboQuant settings based on your model, weight quantization, and hardware. Based on validated Metal testing across M2 Pro and M5 Max.

> **Backend scope:** All validation in this document was performed on Metal (Apple Silicon). CUDA ports exist via community forks but asymmetric q8_0 × turbo has not been independently validated on CUDA.

> **Speed note:** Asymmetric q8_0-K + turbo-V is a quality/robustness rescue, not a speed optimization over q8_0/q8_0. K stays uncompressed, so you trade some decode throughput for quality safety on sensitive models. If symmetric turbo works on your model, use it — it's faster.

## Validated Good

These configurations produce healthy PPL in current testing:

| Model class | Config | Evidence |
|-------------|--------|----------|
| Q8_0 weights, any size | `-ctk turbo4 -ctv turbo4` | phi-4 +1.7%, 35B MoE +0.2%, 27B Dense healthy |
| Q8_0 weights, any size | `-ctk turbo3 -ctv turbo3` | phi-4 +4.2%, 35B MoE +1.1% |
| Q8_0 weights, any size | `-ctk q8_0 -ctv turbo4` | phi-4 +0.3% |
| Q8_0 weights, any size | `-ctk q8_0 -ctv turbo3` | phi-4 +1.1% |
| Q4_K_M, larger models (24B+) | `-ctk turbo3 -ctv turbo3` | Mistral-24B PPL 4.99 (single model tested) |
| Q4_K_M, tested sensitive models | `-ctk q8_0 -ctv turbo4` | Qwen2.5-7B +1.0% |
| Q4_K_M, tested sensitive models | `-ctk q8_0 -ctv turbo3` | Qwen2.5-7B +2.0% |

## Validated Risky

These configurations produce catastrophic PPL in at least one tested model:

| Model class | Config | Evidence |
|-------------|--------|----------|
| Q4_K_M, Qwen2.5-7B | `-ctk turbo4 -ctv turbo4` | PPL 218 (vs 6.58 baseline) |
| Q4_K_M, Qwen2.5-7B | `-ctk turbo3 -ctv turbo3` | PPL 3556 |
| Q4_K_M, Qwen2.5-1.5B | `-ctk turbo3 -ctv turbo3` | PPL 8641+ on both M5 and M2 |

Note: symmetric turbo on Q4_K_M is not universally broken. Mistral-24B Q4_K_M handles it fine. Model family and size both matter.

## Experimental

These configurations showed promising results in current Metal tests but have less validation depth:

| Model class | Config | Evidence |
|-------------|--------|----------|
| Q4_K_M, tested sensitive models | `-ctk q8_0 -ctv turbo2` | Qwen2.5-7B +5.1% — promising but less validated than turbo4-V/turbo3-V |
| Q8_0 weights | `-ctk q8_0 -ctv turbo2` | phi-4 +3.1% — limited testing |

turbo2-V showed promising results in current Metal tests, but has less validation than turbo4-V and turbo3-V. Treat it as experimental.

### Boundary V (experimental, internal mode LA-V7)

A layer-aware V compression strategy that protects boundary layers (first 2 + last 2) with q8_0-V while compressing all remaining layers with turbo2-V. Activated via `TURBO_LAYER_ADAPTIVE=7` env var.

In current Metal testing across 4 models (phi-4-Q8_0, Qwen2.5-7B Q4_K_M, Qwen3.5-35B MoE Q8_0, Qwen3.5-27B Dense Q8_0), Boundary V consistently recovers 37-91% of the turbo2→turbo3 quality gap at effective compression between turbo2 and turbo3 (closer to turbo2 on deeper models). The benefit is largest on deep models where 4 boundary layers is a small fraction of total layers.

| Model | Layers | turbo2 PPL | Boundary V PPL | turbo3 PPL | Quality recovered |
|-------|--------|-----------|---------------|-----------|-------------------|
| phi-4-Q8_0 | 40 | 4.835 | 4.784 | 4.742 | 55% |
| Qwen2.5-7B Q4_K_M | 28 | 6.911 | 6.835 | 6.707 | 37% |
| Qwen3.5-35B MoE | 64 | 5.257 | 5.148 | 5.137 | 91% |
| Qwen3.5-27B Dense | 36 | 6.534 | 6.423 | 6.273 | 42% |

Validated at 512 and 8K context. NIAH retrieval passed. No speed penalty. Effective V bits/val is 2.9-3.4 depending on model depth (between turbo2 and turbo3).

Effective V compression is between turbo2 and turbo3, closer to turbo2 on deeper models.

**How to try it:**

```bash
# Boundary V — boundary layers q8_0-V, rest turbo2-V
TURBO_LAYER_ADAPTIVE=7 llama-server -m model.gguf -ctk q8_0 -ctv turbo2 -fa 1

# Baseline comparison: uniform turbo2-V
llama-server -m model.gguf -ctk q8_0 -ctv turbo2 -fa 1

# Baseline comparison: uniform turbo3-V
llama-server -m model.gguf -ctk q8_0 -ctv turbo3 -fa 1

# PPL validation
TURBO_LAYER_ADAPTIVE=7 llama-perplexity -m model.gguf -ngl 99 -fa 1 \
  -ctk q8_0 -ctv turbo2 -f wikitext-2-raw/wiki.test.raw -c 512 --chunks 4
```

Expected result: PPL better than uniform `q8_0/turbo2`, usually behind uniform `q8_0/turbo3`.

Treat as experimental. Not yet validated at 32K+ context or on CUDA. See [Layer-Aware V Compression](papers/layer-aware-v-compression.md) for the full writeup.

## Recommended Starting Points

| Your situation | Start with | Why |
|---------------|------------|-----|
| Q8_0+ weights | `-ctk turbo4 -ctv turbo4` | Best quality/compression balance |
| Q8_0+ weights, need more compression | `-ctk turbo3 -ctv turbo3` | +4% PPL, 4.6x compression |
| Q4_K_M, unknown model | `-ctk q8_0 -ctv turbo4` | Safe default, V still compressed |
| Q4_K_M, validated large model | `-ctk turbo3 -ctv turbo3` | If you've confirmed PPL is healthy |
| Maximum V compression | `-ctk q8_0 -ctv turbo2` | +5% PPL, experimental (see above) |

**Important framing:** Asymmetric q8_0-K + turbo-V is a **quality/robustness rescue**, not a speed optimization. You trade some decode throughput (K is uncompressed) for quality safety on sensitive models. If your model works fine with symmetric turbo, use symmetric — it's faster.

## Why K Precision Matters More Than V

The attention mechanism computes `softmax(Q * K^T) * V`. K determines which tokens receive attention weight via softmax. Softmax amplifies small errors exponentially: a small shift in Q*K scores can flip which tokens dominate the output. V errors, by contrast, scale linearly through the weighted sum.

In current testing:
- q8_0-K + turbo3-V on Qwen2.5-7B Q4_K_M gives PPL 6.71 (+2.0% vs baseline)
- turbo3-K + q8_0-V on the same model gives PPL 3556 (catastrophic)

Same total bits, opposite directions, 500x quality difference. K precision is the dominant lever.

This is why asymmetric `-ctk q8_0 -ctv turbo3` can rescue models where symmetric `-ctk turbo3 -ctv turbo3` fails. You still get V cache compression while maintaining the attention routing accuracy that K requires.

## Tested Configurations

All results from Metal flash attention on Apple Silicon. PPL measured on wikitext-2-raw (512 context, 4 chunks) unless noted.

### phi-4-14B (Q8_0 weights) — healthy across all configs

| K | V | M5 PPL | M2 PPL | vs q8_0 | Status |
|---|---|--------|--------|---------|--------|
| q8_0 | q8_0 | 4.690 | 4.691 | baseline | healthy |
| turbo4 | turbo4 | 4.770 | 4.787 | +1.7% / +2.0% | healthy |
| turbo3 | turbo3 | 4.886 | 4.956 | +4.2% / +5.7% | healthy |
| q8_0 | turbo4 | 4.702 | 4.693 | +0.3% | healthy |
| q8_0 | turbo3 | 4.742 | — | +1.1% | healthy |
| q8_0 | turbo2 | 4.835 | — | +3.1% | healthy |

Cross-hardware matched: M2 Pro and M5 Max produce equivalent results.

### Qwen2.5-7B-Instruct (Q4_K_M weights) — sensitive to symmetric turbo, rescued by asymmetric K/V

| K | V | M5 PPL | M2 PPL | vs q8_0 | Status |
|---|---|--------|--------|---------|--------|
| q8_0 | q8_0 | 6.577 | 6.579 | baseline | healthy |
| q8_0 | turbo4 | 6.644 | 6.603 | +1.0% | rescued |
| q8_0 | turbo3 | 6.707 | 6.715 | +2.0% | rescued |
| q8_0 | turbo2 | 6.911 | — | +5.1% | rescued |
| turbo4 | turbo4 | 217.7 | 227.5 | catastrophic | avoid |
| turbo3 | turbo3 | 3556 | 3778 | catastrophic | avoid |

Cross-hardware matched: both machines show identical quality patterns.

### Qwen3.5-35B-A3B MoE (Q8_0 weights) — healthy

| K | V | M5 PPL | Status |
|---|---|--------|--------|
| turbo3 | turbo3 | 5.130 | healthy |
| turbo4 | turbo4 | 5.078 | healthy |

### Qwen3.5-27B Dense (Q8_0 weights) — healthy

| K | V | M5 PPL | Status |
|---|---|--------|--------|
| turbo3 | turbo3 | 6.339 | healthy |

### Mistral-Small-24B-Instruct (Q4_K_M weights) — healthy at this size

| K | V | M5 PPL | Status |
|---|---|--------|--------|
| turbo3 | turbo3 | 4.987 | healthy |

This shows Q4_K_M is not universally incompatible with symmetric turbo. The 24B model has enough capacity to absorb the quantization stacking that breaks the 7B model.

## Practical Guidance

### Strong base weights (Q8_0, Q6_K, or higher)

Symmetric turbo works well. Start with turbo4 for best quality, turbo3 for more compression:

```bash
# Best quality with compression
llama-server -m model-Q8_0.gguf -ctk turbo4 -ctv turbo4 -fa 1

# More compression, still healthy
llama-server -m model-Q8_0.gguf -ctk turbo3 -ctv turbo3 -fa 1
```

### Low-bit base weights (Q4_K_M) on sensitive models

In current testing, Qwen2.5-7B Q4_K_M fails catastrophically with symmetric turbo but is rescued by asymmetric K/V. Not all Q4_K_M models are sensitive to this — Mistral-24B Q4_K_M works fine with symmetric turbo. If you're unsure, start with asymmetric:

```bash
# Recommended: near-baseline quality with V compression
llama-server -m model-Q4_K_M.gguf -ctk q8_0 -ctv turbo4 -fa 1

# More V compression, still good (+2% PPL)
llama-server -m model-Q4_K_M.gguf -ctk q8_0 -ctv turbo3 -fa 1

# Maximum V compression (+5% PPL, experimental)
llama-server -m model-Q4_K_M.gguf -ctk q8_0 -ctv turbo2 -fa 1
```

### Low-bit base weights (Q4_K_M) on larger or less sensitive models

In current testing, symmetric turbo3 works on Mistral-24B Q4_K_M (PPL 4.99). Some model families tolerate the quantization stacking. Validate on your specific model, and fall back to asymmetric if quality degrades:

```bash
# Try symmetric first on large Q4_K_M models
llama-server -m large-model-Q4_K_M.gguf -ctk turbo3 -ctv turbo3 -fa 1

# Fall back to asymmetric if quality is poor
llama-server -m large-model-Q4_K_M.gguf -ctk q8_0 -ctv turbo3 -fa 1
```

### Unknown models

Start conservative and validate:

```bash
# Safe starting point for any model
llama-server -m model.gguf -ctk q8_0 -ctv turbo4 -fa 1
```

## Block Size

The default storage block size for turbo3 and turbo2 is 32 elements. Testing with block_size=128 shows identical PPL while improving turbo3 compression from 4.57x to 5.12x.

- **turbo3 (asymmetric q8_0-K + turbo3-V):** Validated across 3 architectures (dense, dense Qwen, hybrid MoE), 3 context lengths (512, 8K, 32K), and 2 hardware platforms (M5 Max, M2 Pro). Zero PPL regression, no speed regression on M5.
- **turbo3 (symmetric turbo3-K + turbo3-V):** Validated on phi-4 dense and Qwen3.5-35B MoE (M5 Max, 512 context). PPL identical at block 32 vs 128.
- **turbo2 (asymmetric q8_0-K + turbo2-V):** Validated on phi-4 Q8_0 (M5) and Qwen2.5-7B Q4_K_M (M5 + M2). PPL identical at block 32 vs 128.
- **NIAH:** 3/3 pass at block_size=128 (phi-4, symmetric turbo3, 4K context). Block128-only run, not strict A/B.

Additionally, on the tested M2 Pro setup (Qwen2.5-1.5B, `q8_0-K + turbo3-V`), block_size=128 improved decode speed by 3-7% across 512 to 16K context. This gain was not observed on M5 Max and may be specific to bandwidth-constrained Apple Silicon. Single-model result; may not generalize.

Community retesting is encouraged on CUDA / non-Apple backends and additional model families once the updated default lands. See [block size study](papers/block-size-experiment.md) for the full data.

## Notes and Caveats

- **Backend scope:** All validation in this document was performed on Metal (Apple Silicon). CUDA ports exist via community forks (spiritbuun, signalnine) but asymmetric q8_0 × turbo has not been independently validated on CUDA.
- **Model sensitivity varies:** The Qwen2.5-7B Q4_K_M failure does not generalize to all Q4_K_M models. Mistral-24B Q4_K_M works fine with symmetric turbo. Test before deploying on new model families.
- **turbo2 as V cache:** turbo2-V showed promising results in current Metal tests (+5.1% PPL on Qwen2.5-7B Q4_K_M, +3.1% on phi-4 Q8_0), but has less validation than turbo4-V and turbo3-V. Treat as experimental.
- **Asymmetric K/V is new:** Mixed q8_0 × turbo kernel support was added in the 2026-03-29 asymmetric K/V update. Earlier builds do not support these configurations.
- **PPL is measured at 512 context with 4 chunks.** Long-context behavior may differ; validate at your target context length.
