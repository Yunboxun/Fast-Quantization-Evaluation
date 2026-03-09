"""
第一层评估：Logit Similarity (Cosine Similarity)
==============================================
目标：快速检测量化模型与原始模型的 logit 分布是否一致
耗时：约 30 秒
原理：对 100 条 prompt，计算量化模型与原模型输出 logit 的 cosine similarity
     相似度越接近 1.0，说明量化质量越高

用法：
    python layer1_logit_similarity.py \
        --ref_model /path/to/original_model \
        --quant_model /path/to/quantized_model \
        --num_prompts 100 \
        --device cuda
"""

import argparse
import time
import json
import os
from typing import List, Optional

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# 默认 100 条评估 prompt（覆盖多种任务类型）
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_PROMPTS = [
    # 通用知识
    "The capital of France is",
    "The largest planet in the solar system is",
    "Water boils at 100 degrees",
    "The speed of light is approximately",
    "The chemical symbol for gold is",
    "Albert Einstein was born in",
    "The human body has approximately",
    "Shakespeare wrote the play",
    "The French Revolution began in",
    "The Great Wall of China was built during",
    # 数学推理
    "If x + 5 = 12, then x =",
    "The square root of 144 is",
    "2 to the power of 10 equals",
    "The area of a circle with radius 5 is",
    "15 multiplied by 7 equals",
    "The derivative of x^2 is",
    "Solve: 3x - 9 = 0. The answer is x =",
    "The sum of angles in a triangle is",
    "25% of 200 is",
    "The Fibonacci sequence starts: 1, 1, 2, 3, 5,",
    # 代码理解
    "def factorial(n):\n    if n == 0:\n        return",
    "In Python, to open a file you use",
    "The time complexity of binary search is",
    "A linked list node typically contains data and a",
    "In object-oriented programming, inheritance means",
    "The keyword 'async' in Python is used for",
    "A database index is used to",
    "The HTTP status code 404 means",
    "In machine learning, overfitting occurs when",
    "A REST API uses HTTP methods like GET, POST,",
    # 语言理解
    "The synonym of 'happy' is",
    "To be or not to be, that is the",
    "The antonym of 'cold' is",
    "She went to the store and bought milk, bread, and",
    "Once upon a time, there was a",
    "The passive voice of 'The cat ate the mouse' is",
    "A metaphor is a figure of speech that",
    "The plural of 'child' is",
    "In English grammar, a noun is",
    "The sentence 'I run every day' is in the",
    # 常识推理
    "If it is raining outside, you should bring an",
    "To make coffee, you need coffee beans, water, and a",
    "Birds can fly because they have",
    "Ice melts when the temperature",
    "A healthy diet includes fruits, vegetables, and",
    "To stay hydrated, you should drink",
    "Exercise is important because it",
    "Sleep is essential for",
    "Reading books can help you",
    "Recycling helps to",
    # 科学知识
    "Photosynthesis is the process by which plants",
    "DNA stands for",
    "The theory of evolution was proposed by",
    "Gravity is a force that",
    "Atoms are made up of protons, neutrons, and",
    "The periodic table organizes elements by",
    "Black holes are regions where",
    "Climate change is primarily caused by",
    "Vaccines work by",
    "The human immune system protects against",
    # 历史地理
    "World War II ended in",
    "The United Nations was founded in",
    "The Amazon River is located in",
    "Mount Everest is the highest peak in",
    "The Renaissance period began in",
    "Columbus reached the Americas in",
    "The Berlin Wall fell in",
    "The first moon landing was in",
    "Ancient Rome was founded by",
    "The Silk Road connected",
    # 推理判断
    "All mammals are warm-blooded. A dog is a mammal. Therefore,",
    "If today is Monday, then tomorrow is",
    "A is taller than B. B is taller than C. Therefore, A is",
    "If all roses are flowers and some flowers fade quickly, then",
    "The more you practice, the better you",
    "If a = b and b = c, then a",
    "Every effect has a",
    "If it's not raining, then I will go for a walk. It's not raining, so",
    "All prime numbers greater than 2 are",
    "If P implies Q, and P is true, then Q is",
    # 技术和 AI
    "Large language models are trained on",
    "The transformer architecture was introduced in the paper",
    "Gradient descent is used to",
    "In neural networks, an activation function",
    "The attention mechanism allows models to",
    "Transfer learning involves",
    "Batch normalization helps to",
    "Dropout in neural networks is used to prevent",
    "The softmax function converts logits to",
    "Backpropagation is used to",
    # 创意和写作
    "A good story needs a compelling",
    "To write a poem, you should start with",
    "The key elements of a persuasive essay are",
    "A haiku has three lines with syllables:",
    "Science fiction often explores themes of",
    "A good introduction should",
    "To describe a sunset, you might write:",
    "The hero's journey typically includes",
    "Dialogue in stories helps to",
    "A plot twist should be surprising yet",
]


# ─────────────────────────────────────────────────────────────────────────────
# 模型加载与推理（统一使用 vLLM）
# ─────────────────────────────────────────────────────────────────────────────

def get_logits_vllm(
    model_path: str,
    prompts: List[str],
    max_tokens: int = 1,
    tensor_parallel_size: int = None,
    dtype: str = "auto",
    gpu_memory_utilization: float = 0.85,
    trust_remote_code: bool = True,
) -> List[torch.Tensor]:
    """
    用 vLLM 加载模型并获取每条 prompt 最后一个 token 的 logprob 分布

    原理：
      vLLM 的 SamplingParams(logprobs=vocab_size) 会返回所有 token 的 log-prob，
      但 vocab_size 可能很大（320xx），改用 prompt_logprobs 来获取 prompt 最后
      一个 token 位置的 logit 分布。

      实际上 vLLM 的 generate() 返回的 logprobs 只覆盖采样到的 token + top-k，
      无法直接获取完整 vocab_size 的 logit vector。

      **解决方案**：使用 vLLM 的 `LLM.encode()` 获取 hidden states，
      或者退而求其次用 prompt_logprobs 的 top-N logprob 来近似。

      这里采用更实用的方案：使用 vLLM 返回的 prompt_logprobs (full vocab)
      来构建稀疏 logit 向量（只保留 top tokens），用于 cosine similarity 计算。

    Args:
        model_path:              量化模型路径
        prompts:                 评估 prompt 列表
        max_tokens:              生成 token 数（设为 1 即可，我们只需要 prompt logits）
        tensor_parallel_size:    张量并行数，None 则自动检测 GPU 数量
        dtype:                   推理精度，"auto" 让 vLLM 自动决定
        gpu_memory_utilization:  GPU 显存利用率上限
        trust_remote_code:       是否信任远程代码

    Returns:
        List of [vocab_size] float32 CPU tensors（稀疏，未出现的 token logit=−∞）
    """
    try:
        from vllm import LLM, SamplingParams
        from vllm.distributed.parallel_state import destroy_model_parallel
    except ImportError:
        raise ImportError(
            "请先安装 vLLM：\n"
            "  pip install vllm\n"
            "或参考: https://docs.vllm.ai/en/latest/getting_started/installation.html"
        )

    # 自动检测 GPU 数量
    if tensor_parallel_size is None:
        tensor_parallel_size = max(1, torch.cuda.device_count())
    print(f"  加载模型 (vLLM): {model_path}")
    print(f"  tensor_parallel_size={tensor_parallel_size}, dtype={dtype}")

    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=trust_remote_code,
        max_model_len=512,      # 限制最大长度，节省显存
        enforce_eager=False,    # 允许 CUDA graph 加速
    )

    # 获取 tokenizer 以便知道 vocab_size
    tokenizer = llm.get_tokenizer()
    vocab_size = len(tokenizer)

    # prompt_logprobs=vocab_size 会返回每个 prompt token 位置的完整 logprob 分布
    # 注意：vLLM 中 prompt_logprobs 的值是 -inf 以外的数，稀疏存储
    # vLLM 默认通常限制 prompt_logprobs 的最大值为 20
    # 为了防止 VLLMValidationError，我们最多只能取 20 个 logprobs
    MAX_VLLM_LOGPROBS = 20
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        prompt_logprobs=MAX_VLLM_LOGPROBS,
        temperature=0.0,
    )

    print(f"  运行推理 ({len(prompts)} prompts)...")
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

    all_logits = []
    for output in outputs:
        # prompt_logprobs: List[Optional[Dict[int, Logprob]]]
        # 每个位置是 token_id -> Logprob 的字典，第 0 位置为 None（BOS 前无 context）
        prompt_lp = output.prompt_logprobs  # 长度 = len(prompt_tokens)
        if prompt_lp is None or len(prompt_lp) == 0:
            # fallback: 返回零向量
            all_logits.append(torch.zeros(vocab_size))
            continue

        # 取最后一个 prompt token 位置的 logprob 分布
        last_pos_lp = prompt_lp[-1]  # Dict[token_id, Logprob] or None
        if last_pos_lp is None:
            all_logits.append(torch.zeros(vocab_size))
            continue

        # 构建 logprob 向量：未返回的 token 给一个极小的默认值（如 -100.0）
        # 避免 -inf 导致 cosine similarity 计算出 NaN
        # 真实的 logprob 一般在 0 到 -30 之间，-100 足够小
        logprob_vec = torch.full((vocab_size,), -100.0)
        for token_id, logprob_obj in last_pos_lp.items():
            if token_id < vocab_size:
                logprob_vec[token_id] = logprob_obj.logprob

        all_logits.append(logprob_vec.float().cpu())

    # 释放 vLLM 显存
    del llm
    try:
        destroy_model_parallel()
    except Exception:
        pass
    torch.cuda.empty_cache()

    return all_logits, tokenizer


def compute_cosine_similarity(
    ref_logits: List[torch.Tensor],
    quant_logits: List[torch.Tensor],
    top_k: int = 20,
) -> dict:
    """
    计算两组 logprob 的 cosine similarity 统计
    
    由于 vLLM 的 logprobs 是稀疏的，直接对几万维的向量（其中大量 -100 填充）算余弦相似度
    会导致相似度虚高（大部分维度都被 -100 支配）。
    因此，这里我们取出 ref 模型输出中 top_k 的 token index，然后仅在这 k 个维度上
    计算 cosine similarity，更能反映真实分布的差异。
    """
    similarities = []
    for ref, quant in zip(ref_logits, quant_logits):
        # 找到原始分布中最重要的 top_k 个 token 位置
        _, topk_indices = torch.topk(ref, min(top_k, ref.shape[0]))
        
        # 提取这 top_k 个位置的 logprob
        ref_topk_vec = ref[topk_indices]
        quant_topk_vec = quant[topk_indices]
        
        # 将 logprob 转换为概率分布 (probability)
        # 这样未命中 top-20 的 -100.0 会变成 0.0，避免在计算余弦相似度时成为巨大的离群值拉低分数
        ref_probs = torch.exp(ref_topk_vec)
        quant_probs = torch.exp(quant_topk_vec)
        
        # 计算在这些重要位置上的 probability cosine similarity
        sim = F.cosine_similarity(ref_probs.unsqueeze(0), quant_probs.unsqueeze(0)).item()
        similarities.append(sim)

    similarities = np.array(similarities)
    return {
        "mean": float(np.mean(similarities)),
        "std": float(np.std(similarities)),
        "min": float(np.min(similarities)),
        "max": float(np.max(similarities)),
        "p25": float(np.percentile(similarities, 25)),
        "p50": float(np.percentile(similarities, 50)),
        "p75": float(np.percentile(similarities, 75)),
        "p10": float(np.percentile(similarities, 10)),  # 最差 10% 的下界
        "below_0_99": float(np.mean(similarities < 0.99)),   # 相似度低于 0.99 的比例
        "below_0_95": float(np.mean(similarities < 0.95)),   # 相似度低于 0.95 的比例
        "below_0_90": float(np.mean(similarities < 0.90)),   # 相似度低于 0.90 的比例
        "per_prompt": similarities.tolist(),
    }


def compute_top_k_agreement(
    ref_logits: List[torch.Tensor],
    quant_logits: List[torch.Tensor],
    k_values: List[int] = [1, 5, 10],
) -> dict:
    """
    计算 top-k token 的一致率
    量化模型的 top-1 token 是否在原模型的 top-k 中
    """
    results = {}
    for k in k_values:
        agreements = []
        for ref, quant in zip(ref_logits, quant_logits):
            ref_topk = torch.topk(ref, k).indices.tolist()
            quant_top1 = torch.argmax(quant).item()
            agreements.append(1 if quant_top1 in ref_topk else 0)
        results[f"top{k}_agreement"] = float(np.mean(agreements))
    return results


def eval_logit_similarity(
    ref_model_path: str,
    quant_model_path: str,
    prompts: Optional[List[str]] = None,
    num_prompts: int = 100,
    max_length: int = 128,
    device: str = "auto",
    dtype: torch.dtype = torch.float16,
    vllm_dtype: str = "auto",
    vllm_gpu_memory_utilization: float = 0.85,
    tensor_parallel_size: int = None,
    output_file: Optional[str] = None,
) -> dict:
    """
    主评估函数：计算量化前后模型的 logprob cosine similarity

    原始模型和量化模型均使用 vLLM 加载。

    Args:
        ref_model_path:                原始模型路径
        quant_model_path:              量化模型路径
        prompts:                       自定义 prompts，None 则使用默认 100 条
        num_prompts:                   使用的 prompt 数量
        max_length:                    未实际使用（vLLM不需要）
        device:                        未实际使用（vLLM自己管设备）
        dtype:                         未实际使用（统一用 vllm_dtype）
        vllm_dtype:                    vLLM 推理精度（"auto"/"float16"/"bfloat16"）
        vllm_gpu_memory_utilization:   vLLM GPU 显存利用率（0.0~1.0）
        tensor_parallel_size:          vLLM 张量并行数，None=自动检测 GPU 数
        output_file:                   结果保存路径（JSON）

    Returns:
        包含所有评估指标的字典
    """
    start_time = time.time()
    print("=" * 60)
    print("第一层评估：Logprob Cosine Similarity (vLLM Compatible)")
    print("=" * 60)

    # 准备 prompts
    if prompts is None:
        prompts = DEFAULT_PROMPTS
    prompts = prompts[:num_prompts]
    print(f"使用 {len(prompts)} 条 prompts")

    # ── [1/4] & [2/4] 用 vLLM 加载原始模型并获取 logits ──
    print("\n[1/4] 加载原始模型 (vLLM) 并计算 logits...")
    ref_logits, ref_tokenizer = get_logits_vllm(
        model_path=ref_model_path,
        prompts=prompts,
        tensor_parallel_size=tensor_parallel_size,
        dtype=vllm_dtype,
        gpu_memory_utilization=vllm_gpu_memory_utilization,
    )
    print(f"  原始模型 logits 获取完成，共 {len(ref_logits)} 条")

    # ── [3/4] & [4/4] 用 vLLM 加载量化模型并获取 logits ──
    print("\n[3/4] 加载量化模型 (vLLM) 并计算 logits...")
    quant_logits, quant_tokenizer = get_logits_vllm(
        model_path=quant_model_path,
        prompts=prompts,
        tensor_parallel_size=tensor_parallel_size,
        dtype=vllm_dtype,
        gpu_memory_utilization=vllm_gpu_memory_utilization,
    )
    print(f"  量化模型 logits 获取完成，共 {len(quant_logits)} 条")

    # 计算指标
    cosine_stats = compute_cosine_similarity(ref_logits, quant_logits)
    topk_stats = compute_top_k_agreement(ref_logits, quant_logits)

    elapsed = time.time() - start_time

    # 汇总结果
    results = {
        "layer": 1,
        "metric": "logit_cosine_similarity",
        "ref_model": ref_model_path,
        "quant_model": quant_model_path,
        "num_prompts": len(prompts),
        "elapsed_seconds": round(elapsed, 2),
        "cosine_similarity": cosine_stats,
        "top_k_agreement": topk_stats,
    }

    # 打印结果
    print("\n" + "=" * 60)
    print("📊 第一层评估结果")
    print("=" * 60)
    print(f"耗时: {elapsed:.1f} 秒")
    print(f"\n── Cosine Similarity ──")
    print(f"  均值 (mean):  {cosine_stats['mean']:.6f}")
    print(f"  标准差 (std): {cosine_stats['std']:.6f}")
    print(f"  最小值 (min): {cosine_stats['min']:.6f}")
    print(f"  P10:          {cosine_stats['p10']:.6f}")
    print(f"  P50 (中位数): {cosine_stats['p50']:.6f}")
    print(f"  最大值 (max): {cosine_stats['max']:.6f}")
    print(f"\n── 质量警告比例 ──")
    print(f"  相似度 < 0.99: {cosine_stats['below_0_99']:.1%}")
    print(f"  相似度 < 0.95: {cosine_stats['below_0_95']:.1%}")
    print(f"  相似度 < 0.90: {cosine_stats['below_0_90']:.1%}")
    print(f"\n── Top-K Token 一致率 ──")
    for k, v in topk_stats.items():
        print(f"  {k}: {v:.1%}")

    # 给出通过/失败建议
    mean_sim = cosine_stats["mean"]
    print(f"\n── 量化质量评级 ──")
    if mean_sim >= 0.999:
        print(f"  ✅ 优秀  (mean cosine ≥ 0.999): {mean_sim:.6f}")
    elif mean_sim >= 0.995:
        print(f"  ✅ 良好  (mean cosine ≥ 0.995): {mean_sim:.6f}")
    elif mean_sim >= 0.990:
        print(f"  ⚠️  一般  (mean cosine ≥ 0.990): {mean_sim:.6f}")
    elif mean_sim >= 0.980:
        print(f"  ⚠️  偏低  (mean cosine ≥ 0.980): {mean_sim:.6f} → 建议做第二层验证")
    else:
        print(f"  ❌ 较差  (mean cosine < 0.980): {mean_sim:.6f} → 量化质量存在问题，建议重新量化")
    print("=" * 60)

    # 保存结果
    if output_file:
        import os
        # 如果传入的是目录，自动补全文件名
        if os.path.isdir(output_file):
            output_file = os.path.join(output_file, "layer1_result.json")
        # per_prompt 数据较大，保存时保留但不在终端打印
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="第一层量化评估：Logprob Cosine Similarity\n"
                    "原始模型和量化模型均使用 vLLM 加载进行推理"
    )
    parser.add_argument(
        "--ref_model", type=str, required=True,
        help="原始（参考）模型路径"
    )
    parser.add_argument(
        "--quant_model", type=str, required=True,
        help="量化模型路径（AWQ/GPTQ/FP8 等量化格式）"
    )
    parser.add_argument(
        "--num_prompts", type=int, default=100,
        help="评估使用的 prompt 数量，默认 100"
    )
    parser.add_argument(
        "--vllm_dtype", type=str, default="auto",
        choices=["auto", "float16", "bfloat16"],
        help="vLLM 量化模型推理精度，默认 auto"
    )
    parser.add_argument(
        "--vllm_gpu_memory_utilization", type=float, default=0.85,
        help="vLLM GPU 显存利用率（0.0~1.0），默认 0.85"
    )
    parser.add_argument(
        "--tensor_parallel_size", type=int, default=None,
        help="vLLM 张量并行 GPU 数量，默认自动检测"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="结果保存路径（JSON 文件）"
    )
    parser.add_argument(
        "--prompts_file", type=str, default=None,
        help="自定义 prompts 文件路径（每行一条 prompt）"
    )
    args = parser.parse_args()

    # 读取自定义 prompts
    custom_prompts = None
    if args.prompts_file:
        with open(args.prompts_file, "r", encoding="utf-8") as f:
            custom_prompts = [line.strip() for line in f if line.strip()]
        print(f"从文件加载了 {len(custom_prompts)} 条 prompts")

    eval_logit_similarity(
        ref_model_path=args.ref_model,
        quant_model_path=args.quant_model,
        prompts=custom_prompts,
        num_prompts=args.num_prompts,
        vllm_dtype=args.vllm_dtype,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()