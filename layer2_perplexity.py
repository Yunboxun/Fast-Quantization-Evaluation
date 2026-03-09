"""
第二层评估：Perplexity（困惑度）
================================
目标：在标准数据集上验证量化模型的语言建模能力损失
耗时：约 5 分钟
数据集：
  - wikitext (wikitext-2-raw-v1 / wikitext-103-raw-v1)
  - c4 (allenai/c4, en 子集)

原理：
  PPL = exp(-1/N * Σ log P(token_i | context))
  PPL 越低越好；量化后 PPL 增量（Δppl）越小说明量化损失越小

业界参考阈值（W4量化）：
  Δppl < 0.1  ── 优秀
  Δppl < 0.5  ── 良好
  Δppl < 1.0  ── 可接受
  Δppl ≥ 1.0  ── 偏高，建议检查量化参数

用法：
    # 只评估量化模型（不对比原模型，速度更快）
    python layer2_perplexity.py \
        --model /path/to/quantized_model \
        --datasets wikitext c4 \
        --device cuda

    # 对比评估（同时评估原模型 + 量化模型，计算 Δppl）
    python layer2_perplexity.py \
        --ref_model /path/to/original_model \
        --model /path/to/quantized_model \
        --datasets wikitext c4 \
        --device cuda
"""

import argparse
import json
import time
import math
from typing import Optional, List, Dict

import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# ─────────────────────────────────────────────────────────────────────────────
# 数据集配置
# ─────────────────────────────────────────────────────────────────────────────
DATASET_CONFIGS = {
    "wikitext2": {
        "path": "wikitext",
        "name": "wikitext-2-raw-v1",
        "split": "test",
        "text_field": "text",
        "join_str": "\n\n",
    },
    "wikitext103": {
        "path": "wikitext",
        "name": "wikitext-103-raw-v1",
        "split": "test",
        "text_field": "text",
        "join_str": "\n\n",
    },
    "c4": {
        "path": "allenai/c4",
        "name": "en",
        "split": "validation",
        "text_field": "text",
        "join_str": "\n\n",
        "max_samples": 1000,   # c4 验证集很大，只取前 1000 条
    },
    "ptb": {
        "path": "ptb_text_only",
        "name": "penn_treebank",
        "split": "test",
        "text_field": "sentence",
        "join_str": "\n",
    },
}

# 别名映射，方便命令行使用
DATASET_ALIASES = {
    "wikitext": "wikitext2",
    "wikitext2": "wikitext2",
    "wikitext103": "wikitext103",
    "c4": "c4",
    "ptb": "ptb",
}


def load_text_dataset(dataset_key: str) -> str:
    """加载数据集并拼接为单一文本字符串"""
    key = DATASET_ALIASES.get(dataset_key, dataset_key)
    if key not in DATASET_CONFIGS:
        raise ValueError(
            f"未知数据集: {dataset_key}，可选: {list(DATASET_ALIASES.keys())}"
        )

    cfg = DATASET_CONFIGS[key]
    print(f"  加载数据集 {cfg['path']} / {cfg.get('name', '')} [{cfg['split']}]...")

    load_kwargs = dict(split=cfg["split"], trust_remote_code=True)
    if cfg.get("name"):
        load_kwargs["name"] = cfg["name"]

    # c4 只取前 N 条以节省时间
    max_samples = cfg.get("max_samples")
    if max_samples:
        load_kwargs["streaming"] = True

    dataset = load_dataset(cfg["path"], **load_kwargs)

    if max_samples:
        # streaming 模式下迭代取前 N 条
        texts = []
        for i, sample in enumerate(dataset):
            if i >= max_samples:
                break
            texts.append(sample[cfg["text_field"]])
    else:
        texts = dataset[cfg["text_field"]]

    full_text = cfg["join_str"].join(texts)
    print(f"  文本总长度: {len(full_text):,} 字符")
    return full_text


@torch.no_grad()
def compute_perplexity(
    model: torch.nn.Module,
    tokenizer,
    text: str,
    seqlen: int = 2048,
    stride: int = 512,
    max_tokens: Optional[int] = None,
    desc: str = "PPL",
) -> dict:
    """
    用滑动窗口方式计算 perplexity

    Args:
        model:      待评估模型
        tokenizer:  分词器
        text:       评估文本
        seqlen:     上下文窗口长度
        stride:     滑动步长（stride < seqlen 时有重叠，更准确）
        max_tokens: 最多处理的 token 数（None=全部）
        desc:       进度条描述

    Returns:
        包含 ppl, nll_mean, num_tokens 的字典
    """
    # Tokenize
    encodings = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = encodings.input_ids[0]  # shape: [total_tokens]

    if max_tokens is not None:
        input_ids = input_ids[:max_tokens]

    total_tokens = input_ids.size(0)
    print(f"  Token 总数: {total_tokens:,}")

    nlls = []
    num_tokens_counted = 0

    # 滑动窗口
    num_windows = max(1, (total_tokens - seqlen + stride - 1) // stride + 1)
    with tqdm(
        range(0, total_tokens - 1, stride),
        total=num_windows,
        desc=desc,
    ) as pbar:
        for begin_loc in pbar:
            end_loc = min(begin_loc + seqlen, total_tokens)
            trg_len = end_loc - max(begin_loc, stride) if begin_loc > 0 else end_loc - begin_loc

            input_chunk = input_ids[begin_loc:end_loc].unsqueeze(0).to(get_model_input_device(model))

            # 构建 labels：只计算 target 部分的 loss
            labels = input_chunk.clone()
            if begin_loc > 0:
                labels[0, : seqlen - trg_len] = -100  # 忽略 context 部分

            outputs = model(input_chunk, labels=labels)
            neg_log_likelihood = outputs.loss * trg_len

            nlls.append(neg_log_likelihood.item())
            num_tokens_counted += trg_len

            # 实时显示当前 PPL
            current_ppl = math.exp(sum(nlls) / num_tokens_counted)
            pbar.set_postfix({"ppl": f"{current_ppl:.3f}"})

            if end_loc == total_tokens:
                break

    mean_nll = sum(nlls) / num_tokens_counted
    ppl = math.exp(mean_nll)

    return {
        "ppl": round(ppl, 4),
        "nll": round(mean_nll, 6),
        "num_tokens": num_tokens_counted,
        "seqlen": seqlen,
        "stride": stride,
    }


def get_model_input_device(model) -> torch.device:
    """获取模型实际所在的第一个参数设备（兼容 device_map='auto'）"""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(
    model_path: str,
    device: str = "auto",
    dtype: torch.dtype = torch.float16,
    trust_remote_code: bool = True,
) -> tuple:
    """加载模型，默认 device_map='auto' 自动多卡分配"""
    print(f"  加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        device_map=device,
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    return model, tokenizer


def eval_perplexity(
    model_path: str,
    ref_model_path: Optional[str] = None,
    datasets: List[str] = None,
    seqlen: int = 2048,
    stride: int = 512,
    device: str = "auto",
    dtype: torch.dtype = torch.float16,
    max_tokens: Optional[int] = None,
    output_file: Optional[str] = None,
) -> dict:
    """
    主评估函数：在多个数据集上计算量化模型 PPL，可选与原模型对比

    Args:
        model_path:     量化模型路径
        ref_model_path: 原始模型路径（可选，用于对比 Δppl）
        datasets:       数据集列表，例如 ["wikitext2", "c4"]
        seqlen:         上下文长度
        stride:         滑动步长
        device:         推理设备
        dtype:          模型精度
        max_tokens:     最多处理的 token 数（None=全部，加速用）
        output_file:    JSON 结果保存路径

    Returns:
        结果字典
    """
    if datasets is None:
        datasets = ["wikitext2", "c4"]

    start_time = time.time()
    print("=" * 60)
    print("第二层评估：Perplexity")
    print("=" * 60)
    print(f"数据集: {datasets}")
    print(f"seqlen={seqlen}, stride={stride}")

    results = {
        "layer": 2,
        "metric": "perplexity",
        "model": model_path,
        "ref_model": ref_model_path,
        "datasets": {},
        "elapsed_seconds": 0,
    }

    # ── 预加载所有数据集文本 ──
    dataset_texts: Dict[str, str] = {}
    for ds in datasets:
        dataset_texts[ds] = load_text_dataset(ds)

    # ── 评估参考模型（若有）──
    ref_results: Dict[str, dict] = {}
    if ref_model_path:
        print(f"\n[阶段 1/2] 评估原始模型 PPL...")
        ref_model, tokenizer = load_model(ref_model_path, device, dtype)
        for ds_name, text in dataset_texts.items():
            print(f"\n  数据集: {ds_name}")
            r = compute_perplexity(
                ref_model, tokenizer, text,
                seqlen=seqlen, stride=stride,
                max_tokens=max_tokens, desc=f"Ref/{ds_name}",
            )
            ref_results[ds_name] = r
            print(f"  原始模型 PPL [{ds_name}]: {r['ppl']:.4f}")
        del ref_model
        if device != "cpu":
            torch.cuda.empty_cache()
        # 重新加载 tokenizer（释放模型后需要）
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = None

    # ── 评估量化模型 ──
    stage_str = "[阶段 2/2]" if ref_model_path else "[评估]"
    print(f"\n{stage_str} 评估量化模型 PPL...")
    quant_model, tokenizer = load_model(model_path, device, dtype)

    for ds_name, text in dataset_texts.items():
        print(f"\n  数据集: {ds_name}")
        r = compute_perplexity(
            quant_model, tokenizer, text,
            seqlen=seqlen, stride=stride,
            max_tokens=max_tokens, desc=f"Quant/{ds_name}",
        )
        print(f"  量化模型 PPL [{ds_name}]: {r['ppl']:.4f}")

        entry = {"quant": r}
        if ds_name in ref_results:
            ref_ppl = ref_results[ds_name]["ppl"]
            delta_ppl = r["ppl"] - ref_ppl
            delta_pct = delta_ppl / ref_ppl * 100
            entry["ref"] = ref_results[ds_name]
            entry["delta_ppl"] = round(delta_ppl, 4)
            entry["delta_pct"] = round(delta_pct, 3)
            print(f"  Δppl = {delta_ppl:+.4f}  ({delta_pct:+.2f}%)")

        results["datasets"][ds_name] = entry

    del quant_model
    if device != "cpu":
        torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    results["elapsed_seconds"] = round(elapsed, 2)

    # ── 打印汇总 ──
    print("\n" + "=" * 60)
    print("📊 第二层评估结果汇总")
    print("=" * 60)
    print(f"耗时: {elapsed:.1f} 秒 ({elapsed/60:.1f} 分钟)")
    print(f"\n{'数据集':<15} {'量化PPL':>10} {'原始PPL':>10} {'Δppl':>10} {'Δ%':>8} {'评级':>8}")
    print("-" * 65)

    all_pass = True
    for ds_name, entry in results["datasets"].items():
        quant_ppl = entry["quant"]["ppl"]
        ref_ppl = entry["ref"]["ppl"] if "ref" in entry else None
        delta = entry.get("delta_ppl")
        delta_pct = entry.get("delta_pct")

        if delta is not None:
            if delta < 0.1:
                grade = "✅ 优秀"
            elif delta < 0.5:
                grade = "✅ 良好"
            elif delta < 1.0:
                grade = "⚠️  可接受"
            else:
                grade = "❌ 偏高"
                all_pass = False
            print(
                f"{ds_name:<15} {quant_ppl:>10.4f} {ref_ppl:>10.4f} "
                f"{delta:>+10.4f} {delta_pct:>+7.2f}% {grade}"
            )
        else:
            print(f"{ds_name:<15} {quant_ppl:>10.4f} {'N/A':>10} {'N/A':>10} {'N/A':>8}")

    if ref_model_path:
        print()
        if all_pass:
            print("✅ 整体评估通过：Δppl 在可接受范围内")
        else:
            print("❌ 注意：部分数据集 Δppl ≥ 1.0，建议检查量化参数")
    print("=" * 60)

    # 保存结果
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="第二层量化评估：Perplexity on WikiText & C4"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="量化模型路径或 HuggingFace model ID"
    )
    parser.add_argument(
        "--ref_model", type=str, default=None,
        help="原始（参考）模型路径，用于计算 Δppl（可选）"
    )
    parser.add_argument(
        "--datasets", nargs="+",
        default=["wikitext2", "c4"],
        choices=list(DATASET_ALIASES.keys()),
        help="评估数据集，可多选。默认: wikitext2 c4"
    )
    parser.add_argument(
        "--seqlen", type=int, default=2048,
        help="上下文窗口长度，默认 2048"
    )
    parser.add_argument(
        "--stride", type=int, default=512,
        help="滑动步长，默认 512（值越小越准确但越慢）"
    )
    parser.add_argument(
        "--max_tokens", type=int, default=None,
        help="最多处理的 token 数（加速评估用，None=全部）"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="推理设备，例如 auto / cuda / cuda:0 / cpu（推荐 auto 自动多卡）"
    )
    parser.add_argument(
        "--dtype", type=str, default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="模型加载精度"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="结果保存路径（JSON 文件）"
    )

    args = parser.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    eval_perplexity(
        model_path=args.model,
        ref_model_path=args.ref_model,
        datasets=args.datasets,
        seqlen=args.seqlen,
        stride=args.stride,
        device=args.device,
        dtype=dtype_map[args.dtype],
        max_tokens=args.max_tokens,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()