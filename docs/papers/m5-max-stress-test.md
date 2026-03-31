# TurboQuant M5 Max 128GB Stress Test — Large Model Validation

**Tom Turney**
Independent Researcher
GitHub: [@TheTom](https://github.com/TheTom)

---

## Abstract

We stress-test TurboQuant KV cache compression on Meta's Llama-3.1-70B-Instruct (Q4_K_M, 40GB) running on a single Apple M5 Max with 128GB unified memory. This is the largest model tested with TurboQuant to date.

Key findings:

1. **Llama-70B Q4_K_M tolerates symmetric turbo quantization.** Unlike Qwen2.5-7B Q4_K_M (catastrophic PPL 3500+ with symmetric turbo3), the 70B model handles all turbo configs without catastrophic failure. turbo4/turbo4 achieves +6.3% PPL vs q8_0, turbo3/turbo3 +11.4%.

2. **turbo3 prefill is faster than q8_0 at 32K context.** 80.8 t/s vs 75.2 t/s (+7.4%). At long context, the KV cache is large enough that turbo3's reduced memory bandwidth outweighs its dequantization cost.

3. **48K context confirmed on a laptop.** 70B at 48K context with turbo3/turbo3 achieves PPL 4.019, using 44GB of 128GB. 61GB remains free.

4. **NIAH retrieval is perfect.** 30/30 single-needle retrieval across 5 depths × 3 context lengths × 2 cache types. Zero difference between q8_0 and turbo3.

5. **Hard context wall at ~49K.** Both q8_0 and turbo3 hang at 50K+ context. This is a Metal backend limitation, not a TurboQuant issue.

All tests used Metal flash attention with full GPU offload. Block size 128 throughout (5.12x compression for turbo3).

---

## 1. Setup

### 1.1 Hardware

| Component | Spec |
|-----------|------|
| SoC | Apple M5 Max |
| Memory | 128GB unified (LPDDR5X) |
| GPU Cores | 40-core |
| Backend | Metal with flash attention |
| OS | macOS |

### 1.2 Model

| Property | Value |
|----------|-------|
| Model | Meta-Llama-3.1-70B-Instruct |
| Weight quantization | Q4_K_M (GGUF, 40GB) |
| Layers | 80 |
| Attention heads | 64 |
| KV heads | 8 (GQA 8:1) |
| Head dimension | 128 |
| Native context | 128K |

### 1.3 Build

- Branch: `feature/turboquant-kv-cache`
- Block size: `QK_TURBO3=128`, `QK_TURBO2=128`
- Sparse V: enabled (default on M5+)
- Boundary V: not active (test predates auto-enable)
- Full GPU offload: `-ngl 99`

---

## 2. Perplexity

### 2.1 Short Context (512 tokens, 20 chunks, wikitext-2-raw)

| K | V | PPL | vs q8_0 | Status |
|---|---|-----|---------|--------|
| q8_0 | q8_0 | 3.257 | baseline | healthy |
| q8_0 | turbo4 | 3.301 | +1.3% | healthy |
| q8_0 | turbo3 | 3.325 | +2.1% | healthy |
| turbo4 | turbo4 | 3.461 | +6.3% | healthy |
| turbo3 | turbo3 | 3.629 | +11.4% | usable |
| turbo2 | turbo2 | 5.161 | +58.5% | degraded |

**Finding:** Llama-70B Q4_K_M tolerates symmetric turbo quantization across all formats. This contrasts with Qwen2.5-7B Q4_K_M, where symmetric turbo3/turbo3 produces catastrophic PPL (3556). The 70B model has sufficient capacity to absorb the quantization stacking that breaks smaller models.

turbo2/turbo2 shows significant degradation (+58.5%) but is not catastrophic — the model remains coherent unlike the Qwen2.5-7B case.

### 2.2 Long Context

| K | V | Context | Chunks | PPL |
|---|---|---------|--------|-----|
| q8_0 | q8_0 | 8K | 4 | 3.617 |
| turbo4 | turbo4 | 8K | 4 | 3.770 |
| turbo3 | turbo3 | 8K | 4 | 3.937 |
| turbo3 | turbo3 | 32K | 2 | 4.839 |
| q8_0 | q8_0 | 48K | 1 | 3.575 |
| turbo3 | turbo3 | 48K | 1 | 4.019 |

PPL remains healthy at all tested context lengths. The turbo3 PPL at 48K (4.019) is higher than q8_0 at 48K (3.575), consistent with the +11.4% gap observed at 512 context.

---

## 3. Speed

### 3.1 Short Context (512 tokens)

| K | V | Prefill (t/s) | Decode (t/s) |
|---|---|:-------------:|:------------:|
| q8_0 | q8_0 | 166.8 | 10.9 |
| q8_0 | turbo4 | 174.4 | 10.1 |
| q8_0 | turbo3 | 174.8 | 10.2 |
| turbo4 | turbo4 | 173.8 | 10.3 |
| turbo3 | turbo3 | 165.0 | 9.9 |
| turbo2 | turbo2 | 170.5 | 9.9 |

Speed is flat across all configs at short context. The 40GB model weights dominate memory bandwidth; the KV cache at 512 tokens is negligible.

### 3.2 8K Context

| K | V | Prefill (t/s) | Decode (t/s) |
|---|---|:-------------:|:------------:|
| q8_0 | q8_0 | 139.2 | 11.9 |
| turbo4 | turbo4 | 134.5 | 10.6 |
| turbo3 | turbo3 | 136.2 | 10.1 |

Still flat. KV cache at 8K is ~1.25GB (q8_0) or ~250MB (turbo3), both negligible vs 40GB weights.

### 3.3 32K Context

| K | V | Prefill (t/s) | Decode (t/s) |
|---|---|:-------------:|:------------:|
| q8_0 | q8_0 | 75.2 | 10.4 |
| turbo4 | turbo4 | 72.5 | 10.5 |
| turbo3 | turbo3 | 80.8 | 10.2 |

**turbo3 prefill is 7.4% faster than q8_0** (80.8 vs 75.2 t/s). At 32K context, the KV cache is large enough (~5GB for q8_0 vs ~1GB for turbo3) that reduced memory bandwidth during attention outweighs dequantization cost. This crossover point is consistent with the observation on smaller models (Qwen3.5-35B MoE on M1 Max, where turbo2 beats q8_0 at 65K prefill).

Decode remains flat at ~10 t/s across all configs. On a 70B model, decode is dominated by the 40GB weight read per token, not the KV cache.

> **Note:** llama-bench with default 5 repetitions hangs on 70B at 32K+ context. All 32K measurements use `-r 1`. Root cause unclear; appears to be Metal resource contention across reps.

---

## 4. Needle-In-A-Haystack (NIAH)

Kamradt single-needle methodology. 5 depths (0%, 25%, 50%, 75%, 100%) × 3 context lengths (4K, 8K, 16K) × 2 cache types.

### q8_0 (baseline)

| Depth | 4K | 8K | 16K |
|-------|:--:|:--:|:---:|
| 0% | PASS | PASS | PASS |
| 25% | PASS | PASS | PASS |
| 50% | PASS | PASS | PASS |
| 75% | PASS | PASS | PASS |
| 100% | PASS | PASS | PASS |

### turbo3 (5.12x compression)

| Depth | 4K | 8K | 16K |
|-------|:--:|:--:|:---:|
| 0% | PASS | PASS | PASS |
| 25% | PASS | PASS | PASS |
| 50% | PASS | PASS | PASS |
| 75% | PASS | PASS | PASS |
| 100% | PASS | PASS | PASS |

**30/30 pass. Zero difference between q8_0 and turbo3.** TurboQuant preserves retrieval accuracy at 5.12x KV cache compression on a 70B model.

---

## 5. Maximum Context

### 5.1 Memory at 48K Context

| Config | KV Cache (MiB) | Model + Context (MiB) | Free (MiB) |
|--------|:--------------:|:---------------------:|:----------:|
| q8_0/q8_0 | 8,160 | 48,991 | 61,108 |
| turbo3/turbo3 | ~3,000 | ~44,000 | ~66,000 |

With turbo3, the KV cache at 48K is 3GB instead of 8GB. Both fit comfortably in 128GB with 60+ GB to spare.

### 5.2 Context Wall

| K | V | Context | Status |
|---|---|---------|--------|
| turbo3 | turbo3 | 48K | works (PPL 4.019) |
| q8_0 | q8_0 | 48K | works (PPL 3.575) |
| turbo3 | turbo3 | 50K | **HANGS** |
| turbo3 | turbo3 | 56K | **HANGS** |
| q8_0 | q8_0 | 64K | **HANGS** |

Both q8_0 and turbo3 hang at 50K+ context. The hang occurs after model loading and KV allocation but before the first computation batch. Memory is not the bottleneck (61GB free at 48K).

This is a Metal backend limitation, not a TurboQuant issue. Preliminary investigation points to Metal compute buffer allocation or residency set handling for large context windows on 70B-class models. See [[TurboQuant - Metal 70B Context Wall Research]] for the full investigation.

---

## 6. GQA Impact on Compression Savings

Llama-3.1-70B uses GQA 8:1 (8 KV heads for 64 attention heads). This means the KV cache is already 1/8th the size it would be without GQA. TurboQuant's compression ratios are the same (5.12x for turbo3 at block_size=128), but the absolute memory savings are smaller:

| Config | KV at 48K | vs fp16 KV |
|--------|:---------:|:----------:|
| fp16 | ~15,360 MiB | 1.0x |
| q8_0 | 8,160 MiB | 1.9x |
| turbo3 | ~3,000 MiB | 5.1x |

On models without GQA (e.g., GPT-class with n_kv_heads = n_heads), TurboQuant's savings scale 8× larger in absolute terms.

---

## 7. Comparison with Smaller Models

| Model | Weights | Symmetric turbo3 PPL | Status |
|-------|---------|:--------------------:|--------|
| Qwen2.5-7B | Q4_K_M | 3,556 | catastrophic |
| Qwen2.5-1.5B | Q4_K_M | 8,641 | catastrophic |
| Mistral-24B | Q4_K_M | 4.987 | healthy |
| **Llama-70B** | **Q4_K_M** | **3.629** | **healthy** |

The Q4_K_M symmetric turbo sensitivity appears to be model-family-dependent, not purely size-dependent. Qwen2.5 is sensitive at all sizes. Mistral and Llama tolerate it. For sensitive models, asymmetric q8_0-K + turbo-V is the recommended path.

---

## 8. Upcoming: Command-R+ 104B Q4_K_M

Testing in progress. Command-R+ 104B (~60GB weights) will be run with two Metal configuration changes that the 70B test did NOT have:

1. `sudo sysctl iogpu.wired_limit_mb=122880` (increase GPU memory cap from default ~96GB to 120GB)
2. `GGML_METAL_NO_RESIDENCY=1` (disable Metal residency sets, suspected cause of 50K+ hang)

These changes allow a direct comparison: if the 104B test pushes past the 49K context wall that blocked the 70B test, the residency set fix is validated. Results will be added to this document.

---

## 9. Limitations

1. **Single model tested.** These results are for Llama-3.1-70B-Instruct Q4_K_M only. Other 70B models (Qwen-72B, DeepSeek-67B) may behave differently.

2. **Q4_K_M weights only.** Q8_0 weights on 70B would likely show even better turbo PPL, but require ~70GB for weights alone, leaving limited room for KV cache on 128GB.

3. **Metal only.** CUDA and HIP backends were not tested at this model size.

4. **Context limited to 48K.** The 50K+ hang prevents testing at the model's native 128K context.

5. **Single run per config.** PPL measurements are single-run, not averaged across multiple seeds. Error bars are from the wikitext-2 chunk variance, not run-to-run variance.

6. **Boundary V not tested.** The auto-enable for turbo2-V was added after this test. Boundary V on 70B turbo2 would likely recover significant quality from the +58.5% degradation.

---

## References

- TurboQuant paper: [arXiv 2504.19874](https://arxiv.org/abs/2504.19874) (ICLR 2026)
- TurboQuant+ implementation: [github.com/TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus)
- llama.cpp fork: [github.com/TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant)
- Block size optimization: [block-size-experiment.md](block-size-experiment.md)
- Sparse V dequant: [sparse-v-dequant.md](sparse-v-dequant.md)
- Configuration recommendations: [turboquant-recommendations.md](../turboquant-recommendations.md)
