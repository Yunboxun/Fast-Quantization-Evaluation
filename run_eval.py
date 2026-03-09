"""
量化模型三层评估流水线 ── 统一入口
======================================
业内标准快速评估组合，从快到慢依次验证量化质量：

  Layer 1 │ Logit Cosine Similarity  │ ~30s   │ 100 条 prompt，cosine 相似度
  Layer 2 │ Perplexity               │ ~5min  │ WikiText-2 + C4 困惑度
  Layer 3 │ Benchmark                │ ~1h+   │ MMLU + GSM8K + HellaSwag

支持三种运行模式：
  --layers 1        只跑第一层（快速筛查）
  --layers 1 2      跑前两层
  --layers 1 2 3    完整三层流水线

用法示例：
    # 完整三层对比评估
    python run_eval.py \
        --ref_model  /path/to/original_model \
        --model      /path/to/quantized_model \
        --layers 1 2 3 \
        --device cuda \
        --output results/eval_report.json

    # 只做第一层快速检测（~30s）
    python run_eval.py \
        --model /path/to/quantized_model \
        --layers 1

    # 快速完整三层（benchmark 使用 limit，总耗时约 10 分钟）
    python run_eval.py \
        --ref_model /path/to/original_model \
        --model     /path/to/quantized_model \
        --layers 1 2 3 \
        --benchmark_limit 200 \
        --device cuda
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import torch

# 确保当前目录可以导入
sys.path.insert(0, str(Path(__file__).parent))

from layer1_logit_similarity import eval_logit_similarity
from layer2_perplexity import eval_perplexity
from layer3_benchmark import eval_benchmarks


# ─────────────────────────────────────────────────────────────────────────────
# 决策阈值配置（决定是否继续下一层）
# ─────────────────────────────────────────────────────────────────────────────
EARLY_STOP_THRESHOLDS = {
    "layer1_cosine_min": 0.95,      # cosine mean 低于此值直接警告
    "layer1_cosine_hard_stop": 0.90, # cosine mean 低于此值可选择终止
    "layer2_delta_ppl_warn": 1.0,    # Δppl 超过此值警告
    "layer2_delta_ppl_stop": 2.0,    # Δppl 超过此值可选择终止
    "layer3_delta_acc_warn": -0.02,  # Δacc 低于此值警告（-2%）
    "layer3_delta_acc_stop": -0.05,  # Δacc 低于此值严重警告（-5%）
}


def print_banner(text: str, width: int = 60):
    print("\n" + "═" * width)
    print(f"  {text}")
    print("═" * width)


def check_layer1_pass(layer1_results: dict, auto_stop: bool = False) -> bool:
    """检查第一层是否通过，返回是否继续"""
    mean_sim = layer1_results["cosine_similarity"]["mean"]
    hard_stop = EARLY_STOP_THRESHOLDS["layer1_cosine_hard_stop"]
    warn = EARLY_STOP_THRESHOLDS["layer1_cosine_min"]

    if mean_sim < hard_stop:
        print(f"\n⚠️  [早停] Cosine Similarity 均值 {mean_sim:.4f} < {hard_stop}，量化质量较差")
        if auto_stop:
            print("   已启用 --auto_stop，跳过后续层评估")
            return False
    elif mean_sim < warn:
        print(f"\n⚠️  [警告] Cosine Similarity 均值 {mean_sim:.4f} < {warn}，建议关注")
    return True


def check_layer2_pass(layer2_results: dict, auto_stop: bool = False) -> bool:
    """检查第二层是否通过，返回是否继续"""
    warn = EARLY_STOP_THRESHOLDS["layer2_delta_ppl_warn"]
    stop = EARLY_STOP_THRESHOLDS["layer2_delta_ppl_stop"]

    max_delta = -float("inf")
    for ds, entry in layer2_results["datasets"].items():
        d = entry.get("delta_ppl")
        if d is not None:
            max_delta = max(max_delta, d)

    if max_delta == -float("inf"):
        return True  # 无对比数据，继续

    if max_delta > stop:
        print(f"\n⚠️  [早停] 最大 Δppl={max_delta:.4f} > {stop}，量化损失较大")
        if auto_stop:
            print("   已启用 --auto_stop，跳过 Benchmark 评估")
            return False
    elif max_delta > warn:
        print(f"\n⚠️  [警告] 最大 Δppl={max_delta:.4f} > {warn}，建议关注")
    return True


def generate_report(
    all_results: dict,
    model_path: str,
    ref_model_path: Optional[str],
    layers_run: List[int],
) -> str:
    """生成人类可读的汇总报告"""
    lines = []
    lines.append("=" * 70)
    lines.append("量化模型评估汇总报告")
    lines.append("=" * 70)
    lines.append(f"评估时间:    {all_results['timestamp']}")
    lines.append(f"量化模型:    {model_path}")
    lines.append(f"参考模型:    {ref_model_path or '未指定（无对比）'}")
    lines.append(f"运行层级:    Layer {', '.join(str(l) for l in layers_run)}")
    lines.append(f"总耗时:      {all_results['total_elapsed_seconds']:.0f}s "
                 f"({all_results['total_elapsed_seconds']/60:.1f}min)")
    lines.append("")

    # Layer 1 摘要
    if "layer1" in all_results:
        r1 = all_results["layer1"]
        cs = r1["cosine_similarity"]
        tk = r1["top_k_agreement"]
        lines.append("┌─ Layer 1: Logit Cosine Similarity")
        lines.append(f"│  耗时:      {r1['elapsed_seconds']}s")
        lines.append(f"│  Prompts:   {r1['num_prompts']} 条")
        lines.append(f"│  Mean Cos:  {cs['mean']:.6f}  (std={cs['std']:.6f})")
        lines.append(f"│  Min Cos:   {cs['min']:.6f}  P10={cs['p10']:.6f}")
        lines.append(f"│  <0.99:     {cs['below_0_99']:.1%}  |  <0.95: {cs['below_0_95']:.1%}  |  <0.90: {cs['below_0_90']:.1%}")
        lines.append(f"│  Top1 Agr:  {tk.get('top1_agreement', 0):.1%}  Top5: {tk.get('top5_agreement', 0):.1%}  Top10: {tk.get('top10_agreement', 0):.1%}")
        mean = cs["mean"]
        if mean >= 0.999:
            grade = "✅ 优秀"
        elif mean >= 0.995:
            grade = "✅ 良好"
        elif mean >= 0.990:
            grade = "⚠️  一般"
        else:
            grade = "❌ 较差"
        lines.append(f"│  评级:      {grade}")
        lines.append("└" + "─" * 50)
        lines.append("")

    # Layer 2 摘要
    if "layer2" in all_results:
        r2 = all_results["layer2"]
        lines.append("┌─ Layer 2: Perplexity")
        lines.append(f"│  耗时:      {r2['elapsed_seconds']}s ({r2['elapsed_seconds']/60:.1f}min)")
        for ds, entry in r2["datasets"].items():
            q_ppl = entry["quant"]["ppl"]
            ref_ppl = entry["ref"]["ppl"] if "ref" in entry else None
            delta = entry.get("delta_ppl")
            if delta is not None:
                d_pct = entry.get("delta_pct", 0)
                if delta < 0.1:
                    g = "✅"
                elif delta < 0.5:
                    g = "✅"
                elif delta < 1.0:
                    g = "⚠️ "
                else:
                    g = "❌"
                lines.append(
                    f"│  {ds:<12} PPL: {q_ppl:.4f} (原={ref_ppl:.4f}, Δ={delta:+.4f}/{d_pct:+.2f}%) {g}"
                )
            else:
                lines.append(f"│  {ds:<12} PPL: {q_ppl:.4f}")
        lines.append("└" + "─" * 50)
        lines.append("")

    # Layer 3 摘要
    if "layer3" in all_results:
        r3 = all_results["layer3"]
        lines.append("┌─ Layer 3: Benchmark")
        lines.append(f"│  耗时:      {r3['elapsed_seconds']}s ({r3['elapsed_seconds']/60:.1f}min)")
        for task, entry in r3["tasks"].items():
            q_score = entry["quant"]["score"]
            ref_score = entry["ref"]["score"] if "ref" in entry else None
            delta = entry.get("delta")
            if delta is not None:
                d_pct = entry.get("delta_pct", 0)
                if delta >= -0.005:
                    g = "✅"
                elif delta >= -0.01:
                    g = "✅"
                elif delta >= -0.02:
                    g = "⚠️ "
                else:
                    g = "❌"
                lines.append(
                    f"│  {task:<16} {q_score:.4f} (原={ref_score:.4f}, Δ={delta:+.4f}/{d_pct:+.2f}%) {g}"
                )
            else:
                score_str = f"{q_score:.4f}" if q_score else "N/A"
                lines.append(f"│  {task:<16} {score_str}")
        lines.append("└" + "─" * 50)
        lines.append("")

    # 综合结论
    lines.append("── 综合结论 ──")
    issues = []

    if "layer1" in all_results:
        mean = all_results["layer1"]["cosine_similarity"]["mean"]
        if mean < 0.990:
            issues.append(f"Layer1 Cosine={mean:.4f} 偏低（建议 ≥ 0.995）")

    if "layer2" in all_results:
        for ds, entry in all_results["layer2"]["datasets"].items():
            d = entry.get("delta_ppl")
            if d and d > 1.0:
                issues.append(f"Layer2 [{ds}] Δppl={d:+.4f} 偏高（建议 < 0.5）")

    if "layer3" in all_results:
        for task, entry in all_results["layer3"]["tasks"].items():
            d = entry.get("delta")
            if d and d < -0.02:
                issues.append(f"Layer3 [{task}] Δacc={d:+.4f} 偏低（建议 > -0.02）")

    if not issues:
        lines.append("✅ 所有层评估通过，量化质量符合标准")
    else:
        lines.append("⚠️  发现以下问题：")
        for issue in issues:
            lines.append(f"   • {issue}")
        lines.append("\n建议：检查量化参数（bits/group_size/awq_scale），考虑调整后重新量化")

    lines.append("=" * 70)
    return "\n".join(lines)


def run_pipeline(
    model_path: str,
    ref_model_path: Optional[str] = None,
    layers: List[int] = None,
    # Layer 1 参数
    num_prompts: int = 100,
    max_length: int = 128,
    # Layer 2 参数
    ppl_datasets: List[str] = None,
    seqlen: int = 2048,
    stride: int = 512,
    max_tokens: Optional[int] = None,
    # Layer 3 参数
    benchmark_tasks: List[str] = None,
    benchmark_batch_size: int = 1,
    benchmark_limit: Optional[int] = None,
    # 公共参数
    device: str = "cuda",
    dtype: str = "float16",
    auto_stop: bool = False,
    output_dir: Optional[str] = None,
) -> dict:
    """
    运行三层评估流水线

    Args:
        model_path:          量化模型路径
        ref_model_path:      原始模型路径（可选）
        layers:              要运行的层级列表，例如 [1, 2, 3]
        num_prompts:         Layer1 prompts 数量
        max_length:          Layer1 输入最大长度
        ppl_datasets:        Layer2 数据集列表
        seqlen:              Layer2 序列长度
        stride:              Layer2 滑动步长
        max_tokens:          Layer2 最大处理 token 数
        benchmark_tasks:     Layer3 benchmark 列表
        benchmark_batch_size:Layer3 batch size
        benchmark_limit:     Layer3 每任务最大样本数
        device:              推理设备
        dtype:               模型精度
        auto_stop:           是否在质量差时自动终止后续层
        output_dir:          输出目录（保存各层 JSON 结果）

    Returns:
        包含所有层结果的汇总字典
    """
    if layers is None:
        layers = [1, 2, 3]
    layers = sorted(set(layers))

    if ppl_datasets is None:
        ppl_datasets = ["wikitext2", "c4"]
    if benchmark_tasks is None:
        benchmark_tasks = ["mmlu", "gsm8k", "hellaswag"]

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    # 准备输出目录
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    pipeline_start = time.time()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    all_results = {
        "timestamp": timestamp,
        "model": model_path,
        "ref_model": ref_model_path,
        "layers_run": layers,
        "total_elapsed_seconds": 0,
    }

    print_banner(f"量化模型三层评估流水线")
    print(f"  量化模型: {model_path}")
    print(f"  原始模型: {ref_model_path or '未指定'}")
    print(f"  运行层级: {layers}")
    print(f"  设备:     {device} | 精度: {dtype}")
    print(f"  自动终止: {'开启' if auto_stop else '关闭'}")

    # ── Layer 1: Logit Similarity ──────────────────────────────────────────
    if 1 in layers:
        print_banner("Layer 1 / 3 — Logit Cosine Similarity  (~30s)")
        out1 = os.path.join(output_dir, "layer1_result.json") if output_dir else None
        r1 = eval_logit_similarity(
            ref_model_path=ref_model_path or model_path,
            quant_model_path=model_path,
            num_prompts=num_prompts,
            max_length=max_length,
            device=device,
            dtype=dtype_map[dtype],
            output_file=out1,
        )
        all_results["layer1"] = r1

        # 早停检查
        if 2 in layers or 3 in layers:
            if not check_layer1_pass(r1, auto_stop=auto_stop):
                layers = [l for l in layers if l == 1]

    # ── Layer 2: Perplexity ────────────────────────────────────────────────
    if 2 in layers:
        print_banner("Layer 2 / 3 — Perplexity on WikiText & C4  (~5min)")
        out2 = os.path.join(output_dir, "layer2_result.json") if output_dir else None
        r2 = eval_perplexity(
            model_path=model_path,
            ref_model_path=ref_model_path,
            datasets=ppl_datasets,
            seqlen=seqlen,
            stride=stride,
            device=device,
            dtype=dtype_map[dtype],
            max_tokens=max_tokens,
            output_file=out2,
        )
        all_results["layer2"] = r2

        # 早停检查
        if 3 in layers:
            if not check_layer2_pass(r2, auto_stop=auto_stop):
                layers = [l for l in layers if l <= 2]

    # ── Layer 3: Benchmark ─────────────────────────────────────────────────
    if 3 in layers:
        print_banner("Layer 3 / 3 — Benchmark (MMLU/GSM8K/HellaSwag)  (~1h+)")
        if benchmark_limit:
            print(f"  ⚡ 快速模式: limit={benchmark_limit} 条/任务")
        out3 = os.path.join(output_dir, "layer3_result.json") if output_dir else None
        r3 = eval_benchmarks(
            model_path=model_path,
            ref_model_path=ref_model_path,
            tasks=benchmark_tasks,
            batch_size=benchmark_batch_size,
            device=device,
            dtype=dtype,
            limit=benchmark_limit,
            output_file=out3,
        )
        all_results["layer3"] = r3

    # ── 总报告 ─────────────────────────────────────────────────────────────
    total_elapsed = time.time() - pipeline_start
    all_results["total_elapsed_seconds"] = round(total_elapsed, 2)
    all_results["layers_actually_run"] = [
        l for l in [1, 2, 3] if f"layer{l}" in all_results
    ]

    report = generate_report(
        all_results,
        model_path=model_path,
        ref_model_path=ref_model_path,
        layers_run=all_results["layers_actually_run"],
    )
    print("\n" + report)

    # 保存总报告
    if output_dir:
        report_json = os.path.join(output_dir, "eval_report.json")
        with open(report_json, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        report_txt = os.path.join(output_dir, "eval_report.txt")
        with open(report_txt, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n📁 报告已保存到目录: {output_dir}")
        print(f"   └── eval_report.json  (结构化数据)")
        print(f"   └── eval_report.txt   (可读报告)")
        for l in all_results["layers_actually_run"]:
            print(f"   └── layer{l}_result.json")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="量化模型三层评估流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 完整三层对比评估
  python run_eval.py --ref_model /path/fp16 --model /path/w4 --layers 1 2 3

  # 只跑第一层快速检测（~30s）
  python run_eval.py --model /path/w4 --layers 1

  # 快速完整三层（benchmark 用 200 条样本，约 10 分钟）
  python run_eval.py --ref_model /path/fp16 --model /path/w4 \\
      --layers 1 2 3 --benchmark_limit 200 --output_dir ./eval_results

  # 评估大模型（bfloat16，多卡）
  python run_eval.py --ref_model /path/fp16 --model /path/w4 \\
      --layers 1 2 3 --dtype bfloat16 --device auto \\
      --benchmark_batch_size 4 --output_dir ./eval_results
        """
    )

    # 模型路径
    parser.add_argument("--model", type=str, required=True,
                        help="量化模型路径")
    parser.add_argument("--ref_model", type=str, default=None,
                        help="原始（FP16）模型路径（可选，用于计算差值）")

    # 流水线控制
    parser.add_argument("--layers", nargs="+", type=int, default=[1, 2, 3],
                        choices=[1, 2, 3],
                        help="运行的层级，默认 1 2 3")
    parser.add_argument("--auto_stop", action="store_true",
                        help="当检测到量化质量差时自动跳过后续层")

    # Layer 1 参数
    parser.add_argument("--num_prompts", type=int, default=100,
                        help="Layer1: 评估 prompt 数量，默认 100")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Layer1: 输入最大 token 数，默认 128")

    # Layer 2 参数
    parser.add_argument("--ppl_datasets", nargs="+",
                        default=["wikitext2", "c4"],
                        choices=["wikitext2", "wikitext103", "c4", "ptb"],
                        help="Layer2: 数据集，默认 wikitext2 c4")
    parser.add_argument("--seqlen", type=int, default=2048,
                        help="Layer2: 序列长度，默认 2048")
    parser.add_argument("--stride", type=int, default=512,
                        help="Layer2: 滑动步长，默认 512")
    parser.add_argument("--max_tokens", type=int, default=None,
                        help="Layer2: 最多处理 token 数（加速用）")

    # Layer 3 参数
    parser.add_argument("--benchmark_tasks", nargs="+",
                        default=["mmlu", "gsm8k", "hellaswag"],
                        help="Layer3: benchmark 任务，默认 mmlu gsm8k hellaswag")
    parser.add_argument("--benchmark_batch_size", type=int, default=1,
                        help="Layer3: batch size，默认 1")
    parser.add_argument("--benchmark_limit", type=int, default=None,
                        help="Layer3: 每任务最多样本数（快速验证用）")

    # 公共参数
    parser.add_argument("--device", type=str, default="cuda",
                        help="推理设备，例如 cuda / cuda:0 / auto / cpu")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "bfloat16", "float32"],
                        help="模型加载精度，默认 float16")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="结果保存目录（保存 JSON + txt 报告）")

    args = parser.parse_args()

    run_pipeline(
        model_path=args.model,
        ref_model_path=args.ref_model,
        layers=args.layers,
        num_prompts=args.num_prompts,
        max_length=args.max_length,
        ppl_datasets=args.ppl_datasets,
        seqlen=args.seqlen,
        stride=args.stride,
        max_tokens=args.max_tokens,
        benchmark_tasks=args.benchmark_tasks,
        benchmark_batch_size=args.benchmark_batch_size,
        benchmark_limit=args.benchmark_limit,
        device=args.device,
        dtype=args.dtype,
        auto_stop=args.auto_stop,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()