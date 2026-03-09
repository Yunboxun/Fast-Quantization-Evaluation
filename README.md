# Fast Quantization Evaluation（量化模型三层快速评估）

业内常见的快速评估组合，用于验证 LLM 量化后的质量损失。从快到慢分三层递进检测，任何一层发现问题都可提前中止，节省评估成本。

---

## 三层评估体系

| 层级 | 方法 | 数据集/任务 | 耗时 | 作用 |
|:----:|------|------------|:----:|------|
| **Layer 1** | Logit Cosine Similarity | 100 条 prompt | ~30s | 快速筛查 logit 分布偏移 |
| **Layer 2** | Perplexity | WikiText-2 + C4 | ~5min | 验证语言建模能力损失 |
| **Layer 3** | Benchmark | MMLU + GSM8K + HellaSwag | ~1h+ | 验证下游任务能力 |

---

## 文件结构

```
Fast-Quantization-Evaluation/
├── run_eval.py                # ✅ 统一入口（推荐使用）
├── layer1_logit_similarity.py # 第一层：Logit Cosine Similarity
├── layer2_perplexity.py       # 第二层：Perplexity（困惑度）
├── layer3_benchmark.py        # 第三层：标准 Benchmark
└── README.md
```

---

## 安装依赖

```bash
pip install torch transformers datasets tqdm
pip install lm-eval        # Layer 3 需要
```

---

## 快速开始

### 完整三层对比评估（推荐）

```bash
python run_eval.py \
    --ref_model  /path/to/original_fp16_model \
    --model      /path/to/quantized_model \
    --layers 1 2 3 \
    --device cuda \
    --output_dir ./eval_results
```

### 只做第一层快速筛查（~30秒）

```bash
python run_eval.py \
    --model /path/to/quantized_model \
    --layers 1 \
    --device cuda
```

### 快速完整评估（benchmark 用 200 条样本，约 10 分钟）

```bash
python run_eval.py \
    --ref_model /path/to/original_fp16_model \
    --model     /path/to/quantized_model \
    --layers 1 2 3 \
    --benchmark_limit 200 \
    --device cuda \
    --output_dir ./eval_results
```

### 大模型（70B+）推荐配置

```bash
python run_eval.py \
    --ref_model /path/to/original_fp16_model \
    --model     /path/to/quantized_model \
    --layers 1 2 3 \
    --dtype bfloat16 \
    --device auto \
    --benchmark_batch_size 4 \
    --max_tokens 50000 \
    --output_dir ./eval_results
```

---

## 各层详细说明

### Layer 1：Logit Cosine Similarity

**原理**：对每条 prompt，取最后一个 token 位置的 logit 向量，计算量化模型与原模型的余弦相似度。

```
similarity = cos(logit_ref, logit_quant)
```

**质量阈值**：

| 均值 | 评级 |
|------|------|
| ≥ 0.999 | ✅ 优秀 |
| ≥ 0.995 | ✅ 良好 |
| ≥ 0.990 | ⚠️ 一般 |
| ≥ 0.980 | ⚠️ 偏低 |
| < 0.980 | ❌ 较差 |

**单独运行**：
```bash
python layer1_logit_similarity.py \
    --ref_model /path/to/fp16 \
    --quant_model /path/to/quantized \
    --num_prompts 100 \
    --device cuda \
    --output layer1_result.json
```

---

### Layer 2：Perplexity

**原理**：用滑动窗口方式在标准数据集上计算语言模型困惑度。

```
PPL = exp(-1/N * Σ log P(token_i | context))
```

**支持的数据集**：`wikitext2`、`wikitext103`、`c4`、`ptb`

**质量阈值（W4 量化参考）**：

| Δppl | 评级 |
|------|------|
| < 0.1 | ✅ 优秀 |
| < 0.5 | ✅ 良好 |
| < 1.0 | ⚠️ 可接受 |
| ≥ 1.0 | ❌ 偏高 |

**单独运行**：
```bash
python layer2_perplexity.py \
    --model /path/to/quantized \
    --ref_model /path/to/fp16 \
    --datasets wikitext2 c4 \
    --seqlen 2048 \
    --device cuda \
    --output layer2_result.json
```

---

### Layer 3：Benchmark

基于 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) 在标准 NLP benchmark 上评估。

**支持的 Benchmark**：

| 名称 | few-shot | 指标 | 说明 |
|------|:--------:|------|------|
| `mmlu` ★ | 5 | acc | 多任务语言理解，57 个学科 |
| `gsm8k` ★ | 8 | exact_match | 小学数学推理（CoT） |
| `hellaswag` ★ | 10 | acc_norm | 常识推理完型填空 |
| `arc_easy` | 25 | acc_norm | ARC 小学科学题（简单） |
| `arc_challenge` | 25 | acc_norm | ARC 小学科学题（挑战） |
| `truthfulqa_mc` | 0 | acc | 真实性评估 |
| `winogrande` | 5 | acc | 常识推理 |

★ = 默认运行的三个核心任务

**质量阈值**：

| Δacc | 评级 |
|------|------|
| ≥ -0.5% | ✅ 优秀 |
| ≥ -1.0% | ✅ 良好 |
| ≥ -2.0% | ⚠️ 可接受 |
| ≥ -5.0% | ⚠️ 偏低 |
| < -5.0% | ❌ 严重退化 |

**单独运行**：
```bash
# 列出所有可用 benchmark
python layer3_benchmark.py --list_tasks

# 运行评估
python layer3_benchmark.py \
    --model /path/to/quantized \
    --ref_model /path/to/fp16 \
    --tasks mmlu gsm8k hellaswag \
    --batch_size 4 \
    --device cuda \
    --output layer3_result.json
```

---

## 统一入口参数说明

```
必填参数:
  --model               量化模型路径
  --ref_model           原始 FP16 模型路径（可选，用于计算 Δ 差值）

流水线控制:
  --layers              运行层级，默认 1 2 3
  --auto_stop           质量差时自动跳过后续层

Layer 1 参数:
  --num_prompts         评估 prompt 数量，默认 100
  --max_length          输入最大 token 数，默认 128

Layer 2 参数:
  --ppl_datasets        数据集，默认 wikitext2 c4
  --seqlen              序列长度，默认 2048
  --stride              滑动步长，默认 512
  --max_tokens          最多处理 token 数（加速用）

Layer 3 参数:
  --benchmark_tasks     任务列表，默认 mmlu gsm8k hellaswag
  --benchmark_batch_size  batch size，默认 1
  --benchmark_limit     每任务最多样本数（快速验证用）

公共参数:
  --device              推理设备，默认 cuda
  --dtype               精度，默认 float16（可选 bfloat16 float32）
  --output_dir          结果保存目录
```

---

## 决策流程

```
量化完成
    │
    ▼
Layer 1 (30s)  ── cosine < 0.95? ──► ❌ 量化失败，重新调参
    │
    ▼ cosine ≥ 0.95，继续
Layer 2 (5min) ── Δppl > 2.0?   ──► ❌ 损失过大，重新调参
    │
    ▼ Δppl ≤ 2.0，继续
Layer 3 (1h+)  ── Δacc < -5%?   ──► ❌ 能力退化，重新调参
    │
    ▼ 全部通过
✅ 量化质量达标，可以发布
```

---

## 输出文件

指定 `--output_dir` 后生成：

```
eval_results/
├── eval_report.json      # 结构化结果
├── eval_report.txt       # 可读汇总报告
├── layer1_result.json
├── layer2_result.json
└── layer3_result.json
```

---

## 参考

- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [MMLU](https://arxiv.org/abs/2009.03300)
- [GSM8K](https://arxiv.org/abs/2110.14168)
- [HellaSwag](https://arxiv.org/abs/1905.07830)
- [AutoAWQ](https://github.com/casper-hansen/AutoAWQ)
- [llm-compressor](https://github.com/vllm-project/llm-compressor)