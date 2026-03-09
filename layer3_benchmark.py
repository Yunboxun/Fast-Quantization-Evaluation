"""
第三层评估：Benchmark（标准能力评估）
======================================
目标：用业界标准 benchmark 量化模型能力损失
耗时：1 小时+（取决于模型大小和硬件）

支持的 Benchmark：
  - mmlu        多任务语言理解（57 个学科，5-shot）
  - gsm8k       小学数学推理（8-shot，CoT）
  - hellaswag   常识推理完型（10-shot）
  - arc_easy    ARC 小学科学题（简单）
  - arc_challenge ARC 小学科学题（挑战）
  - truthfulqa_mc TruthfulQA 多选（0-shot）

依赖：
    pip install lm-eval  # lm-evaluation-harness >= 0.4.0

用法：
    # 评估量化模型
    python layer3_benchmark.py \
        --model /path/to/quantized_model \
        --tasks mmlu gsm8k hellaswag \
        --device cuda \
        --batch_size 8

    # 对比评估（量化 vs 原始）
    python layer3_benchmark.py \
        --ref_model /path/to/original_model \
        --model /path/to/quantized_model \
        --tasks mmlu gsm8k hellaswag \
        --device cuda \
        --batch_size 8

    # 快速验证（每个 benchmark 只取子集样本）
    python layer3_benchmark.py \
        --model /path/to/quantized_model \
        --tasks mmlu gsm8k hellaswag \
        --limit 100 \
        --device cuda
"""

import argparse
import json
import time
from typing import Optional, List, Dict

import torch


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark 配置
# ─────────────────────────────────────────────────────────────────────────────
BENCHMARK_CONFIGS = {
    "mmlu": {
        "task_name": "mmlu",            # lm_eval 任务名
        "num_fewshot": 5,
        "metric": "acc",
        "description": "多任务语言理解 (MMLU, 57 subjects)",
        "ref_scores": {                  # 常见模型参考分数，方便对比
            "llama2-7b": 0.458,
            "llama2-13b": 0.548,
            "llama3-8b": 0.661,
            "qwen2.5-7b": 0.741,
        },
    },
    "gsm8k": {
        "task_name": "gsm8k",
        "num_fewshot": 8,
        "metric": "exact_match",
        "description": "小学数学推理 (GSM8K, CoT)",
        "ref_scores": {
            "llama2-7b": 0.142,
            "llama2-13b": 0.291,
            "llama3-8b": 0.756,
            "qwen2.5-7b": 0.852,
        },
    },
    "hellaswag": {
        "task_name": "hellaswag",
        "num_fewshot": 10,
        "metric": "acc_norm",
        "description": "常识推理完型填空 (HellaSwag)",
        "ref_scores": {
            "llama2-7b": 0.757,
            "llama2-13b": 0.802,
            "llama3-8b": 0.821,
            "qwen2.5-7b": 0.806,
        },
    },
    "arc_easy": {
        "task_name": "arc_easy",
        "num_fewshot": 25,
        "metric": "acc_norm",
        "description": "ARC 小学科学题（简单）",
        "ref_scores": {},
    },
    "arc_challenge": {
        "task_name": "arc_challenge",
        "num_fewshot": 25,
        "metric": "acc_norm",
        "description": "ARC 小学科学题（挑战）",
        "ref_scores": {},
    },
    "truthfulqa_mc": {
        "task_name": "truthfulqa_mc2",
        "num_fewshot": 0,
        "metric": "acc",
        "description": "TruthfulQA 真实性评估",
        "ref_scores": {},
    },
    "winogrande": {
        "task_name": "winogrande",
        "num_fewshot": 5,
        "metric": "acc",
        "description": "Winogrande 常识推理",
        "ref_scores": {},
    },
}

# 默认运行的三个核心 benchmark
DEFAULT_TASKS = ["mmlu", "gsm8k", "hellaswag"]


def get_lm_eval_model(
    model_path: str,
    device: str = "auto",
    dtype: str = "float16",
    batch_size: int = 1,
):
    """
    构建 lm_eval 兼容的模型对象
    支持 HuggingFace transformers 模型
    """
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        raise ImportError(
            "请安装 lm-evaluation-harness:\n"
            "  pip install lm-eval\n"
            "或从源码安装:\n"
            "  pip install git+https://github.com/EleutherAI/lm-evaluation-harness.git"
        )

    print(f"  加载模型: {model_path}")
    lm_model = HFLM(
        pretrained=model_path,
        dtype=dtype,
        device=device,
        batch_size=batch_size,
        trust_remote_code=True,
    )
    return lm_model


def run_single_benchmark(
    lm_model,
    task_key: str,
    limit: Optional[int] = None,
    num_fewshot: Optional[int] = None,
) -> dict:
    """
    运行单个 benchmark，返回评估结果

    Args:
        lm_model:    lm_eval 模型对象
        task_key:    benchmark 名称（见 BENCHMARK_CONFIGS）
        limit:       每个任务最多评估的样本数（None=全部）
        num_fewshot: 覆盖默认的 few-shot 数量

    Returns:
        包含 score 和原始 results 的字典
    """
    try:
        import lm_eval
        from lm_eval import evaluator
    except ImportError:
        raise ImportError("请安装 lm-eval: pip install lm-eval")

    cfg = BENCHMARK_CONFIGS[task_key]
    n_shot = num_fewshot if num_fewshot is not None else cfg["num_fewshot"]
    task_name = cfg["task_name"]
    metric = cfg["metric"]

    print(f"\n  运行 {task_key} ({cfg['description']}, {n_shot}-shot)...")
    t0 = time.time()

    # lm_eval >= 0.4.x API
    results = lm_eval.simple_evaluate(
        model=lm_model,
        tasks=[task_name],
        num_fewshot=n_shot,
        limit=limit,
        log_samples=False,
    )

    elapsed = time.time() - t0

    # 提取分数
    task_results = results["results"].get(task_name, {})
    # 尝试找到指定 metric
    score = None
    for key in [metric, f"{metric},none", f"{metric}_stderr,none"]:
        if key in task_results and not key.endswith("stderr,none"):
            score = task_results[key]
            break

    # 如果没找到，取第一个数值字段
    if score is None:
        for k, v in task_results.items():
            if isinstance(v, float) and not k.endswith("stderr"):
                score = v
                metric = k
                break

    return {
        "task": task_key,
        "task_name": task_name,
        "num_fewshot": n_shot,
        "metric": metric,
        "score": round(score, 6) if score is not None else None,
        "elapsed_seconds": round(elapsed, 2),
        "raw": task_results,
    }


def eval_benchmarks(
    model_path: str,
    ref_model_path: Optional[str] = None,
    tasks: List[str] = None,
    batch_size: int = 1,
    device: str = "auto",
    dtype: str = "float16",
    limit: Optional[int] = None,
    num_fewshot: Optional[int] = None,
    output_file: Optional[str] = None,
) -> dict:
    """
    主评估函数：在多个 benchmark 上评估量化模型，可选与原模型对比

    Args:
        model_path:     量化模型路径
        ref_model_path: 原始模型路径（可选，对比用）
        tasks:          benchmark 列表，None 则使用默认 [mmlu, gsm8k, hellaswag]
        batch_size:     推理 batch size
        device:         推理设备
        dtype:          模型精度
        limit:          每个任务最多评估样本数（快速验证用）
        num_fewshot:    覆盖默认 few-shot 设置
        output_file:    结果保存路径

    Returns:
        评估结果字典
    """
    if tasks is None:
        tasks = DEFAULT_TASKS

    # 验证所有任务名称有效
    unknown = [t for t in tasks if t not in BENCHMARK_CONFIGS]
    if unknown:
        raise ValueError(
            f"未知 benchmark: {unknown}\n"
            f"可选: {list(BENCHMARK_CONFIGS.keys())}"
        )

    start_time = time.time()
    print("=" * 60)
    print("第三层评估：Benchmark")
    print("=" * 60)
    print(f"任务: {tasks}")
    print(f"batch_size={batch_size}, device={device}, dtype={dtype}")
    if limit:
        print(f"⚠️  快速模式: 每个任务最多 {limit} 条样本")

    results = {
        "layer": 3,
        "metric": "benchmark_accuracy",
        "model": model_path,
        "ref_model": ref_model_path,
        "tasks": {},
        "elapsed_seconds": 0,
    }

    # ── 评估参考模型（若有）──
    ref_task_results: Dict[str, dict] = {}
    if ref_model_path:
        print(f"\n[阶段 1/2] 评估原始模型...")
        ref_lm = get_lm_eval_model(ref_model_path, device, dtype, batch_size)
        for task in tasks:
            r = run_single_benchmark(ref_lm, task, limit=limit, num_fewshot=num_fewshot)
            ref_task_results[task] = r
            print(f"  {task}: {r['score']:.4f} ({r['elapsed_seconds']:.0f}s)")
        del ref_lm
        if device != "cpu":
            torch.cuda.empty_cache()

    # ── 评估量化模型 ──
    stage_str = "[阶段 2/2]" if ref_model_path else "[评估]"
    print(f"\n{stage_str} 评估量化模型...")
    quant_lm = get_lm_eval_model(model_path, device, dtype, batch_size)

    for task in tasks:
        r = run_single_benchmark(quant_lm, task, limit=limit, num_fewshot=num_fewshot)
        print(f"  {task}: {r['score']:.4f} ({r['elapsed_seconds']:.0f}s)")

        entry = {"quant": r}
        if task in ref_task_results:
            ref_score = ref_task_results[task]["score"]
            quant_score = r["score"]
            if ref_score and quant_score:
                delta = quant_score - ref_score
                delta_pct = delta / ref_score * 100
                entry["ref"] = ref_task_results[task]
                entry["delta"] = round(delta, 6)
                entry["delta_pct"] = round(delta_pct, 3)
                print(f"  Δacc = {delta:+.4f}  ({delta_pct:+.2f}%)")

        results["tasks"][task] = entry

    del quant_lm
    if device != "cpu":
        torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    results["elapsed_seconds"] = round(elapsed, 2)

    # ── 打印汇总 ──
    print("\n" + "=" * 60)
    print("📊 第三层评估结果汇总")
    print("=" * 60)
    total_min = elapsed / 60
    print(f"耗时: {elapsed:.0f} 秒 ({total_min:.1f} 分钟)")
    if limit:
        print(f"⚠️  注意：使用了 limit={limit}，结果仅供快速参考，非完整评估")

    print(f"\n{'Benchmark':<20} {'量化模型':>10} {'原始模型':>10} {'Δacc':>10} {'Δ%':>8} {'评级':>10}")
    print("-" * 72)

    all_pass = True
    for task, entry in results["tasks"].items():
        cfg = BENCHMARK_CONFIGS[task]
        quant_score = entry["quant"]["score"]
        ref_score = entry["ref"]["score"] if "ref" in entry else None
        delta = entry.get("delta")
        delta_pct = entry.get("delta_pct")

        if delta is not None:
            # 量化后精度下降评级
            if delta >= -0.005:          # 下降 < 0.5%
                grade = "✅ 优秀"
            elif delta >= -0.01:         # 下降 1%
                grade = "✅ 良好"
            elif delta >= -0.02:         # 下降 2%
                grade = "⚠️  可接受"
            elif delta >= -0.05:         # 下降 5%
                grade = "⚠️  偏低"
                all_pass = False
            else:                        # 下降 > 5%
                grade = "❌ 严重退化"
                all_pass = False

            desc = cfg["description"][:18]
            print(
                f"{task:<20} {quant_score:>10.4f} {ref_score:>10.4f} "
                f"{delta:>+10.4f} {delta_pct:>+7.2f}% {grade}"
            )
        else:
            quant_str = f"{quant_score:.4f}" if quant_score else "N/A"
            desc = cfg["description"][:18]
            print(f"{task:<20} {quant_str:>10} {'N/A':>10} {'N/A':>10} {'N/A':>8}")

    if ref_model_path:
        print()
        if all_pass:
            print("✅ 整体评估通过：benchmark 精度下降在可接受范围内")
        else:
            print("❌ 注意：部分 benchmark 精度下降超过阈值，建议排查量化参数")

    print("\n── 评级说明 ──")
    print("  ✅ 优秀：Δacc < -0.5%")
    print("  ✅ 良好：Δacc < -1.0%")
    print("  ⚠️  可接受：Δacc < -2.0%")
    print("  ⚠️  偏低：Δacc < -5.0%")
    print("  ❌ 严重退化：Δacc ≥ -5.0%")
    print("=" * 60)

    # 保存结果
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {output_file}")

    return results


def list_available_tasks():
    """打印所有可用的 benchmark 信息"""
    print("\n可用 Benchmark 列表:")
    print(f"{'名称':<20} {'few-shot':>8} {'指标':<18} {'描述'}")
    print("-" * 70)
    for name, cfg in BENCHMARK_CONFIGS.items():
        is_default = "★" if name in DEFAULT_TASKS else " "
        print(
            f"{is_default} {name:<18} {cfg['num_fewshot']:>8}-shot  "
            f"{cfg['metric']:<16} {cfg['description']}"
        )
    print("\n★ = 默认运行")


def main():
    parser = argparse.ArgumentParser(
        description="第三层量化评估：Benchmark (MMLU / GSM8K / HellaSwag)"
    )
    parser.add_argument(
        "--model", type=str, required=False,
        help="量化模型路径或 HuggingFace model ID"
    )
    parser.add_argument(
        "--ref_model", type=str, default=None,
        help="原始（参考）模型路径，用于计算 Δacc（可选）"
    )
    parser.add_argument(
        "--tasks", nargs="+",
        default=DEFAULT_TASKS,
        help=f"评估任务，可多选。默认: {DEFAULT_TASKS}"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="推理 batch size，默认 1"
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
        "--limit", type=int, default=None,
        help="每个任务最多评估样本数（快速验证用，None=全部）"
    )
    parser.add_argument(
        "--num_fewshot", type=int, default=None,
        help="覆盖默认 few-shot 数量"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="结果保存路径（JSON 文件）"
    )
    parser.add_argument(
        "--list_tasks", action="store_true",
        help="列出所有可用的 benchmark"
    )

    args = parser.parse_args()

    if args.list_tasks:
        list_available_tasks()
        return

    if not args.model:
        parser.error("--model 是必填参数（除非使用 --list_tasks）")

    eval_benchmarks(
        model_path=args.model,
        ref_model_path=args.ref_model,
        tasks=args.tasks,
        batch_size=args.batch_size,
        device=args.device,
        dtype=args.dtype,
        limit=args.limit,
        num_fewshot=args.num_fewshot,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()