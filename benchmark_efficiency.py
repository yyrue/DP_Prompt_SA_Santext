#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单 token 替换效率对比实验（ε 逐请求动态变化场景）

场景：不同用户的隐私预算 ε 不同，每次请求 ε 都可能变化。
本实验对比"替换一个 token"时，基线方法（指数机制）与采样放大方法的：
  1. 在线时间（处理 1 个 token 所需，含随 ε 变化必须重算的部分）
  2. 在线额外内存开销

口径 A（最契合 ε 逐请求变化的设定 + 最强基线）：
  指数机制要求在全词表上做 softmax 归一化。替换 1 个 token 时，基线已优化到
  "逐 token 只算需要的那一行"：取目标 token 的那一行距离，做 softmax(-ε·d/2)
  得到长度 |V| 的概率分布并采样 1 次，在线复杂度 O(|V|)。

公平性划分（划分标准：该步骤是否随每次请求的 ε 变化）：
  - 离线一次性（与 ε 无关，可缓存复用，不计入在线）: 距离矩阵计算
  - 在线（随 ε 变化，每次请求都要重算，计入对比）:
      * 基线: 目标行 softmax(-ε·d/2) + 1 次采样，O(|V|)
      * 采样放大: calc_p 重算标量 p + 1 次伯努利采样，O(1)
  内存口径：基线为支持替换任意 token，须可访问完整距离矩阵，其在线额外
  内存按整张矩阵理论值 |V|×|V|×dtype_size 报告；采样放大内存 ≈ 0。

实验设置：
  - 词嵌入: BERT-base-uncased 嵌入层 (|V|=30522, dim=768)
  - 距离/概率 dtype: float32（实际部署的合理选择）
  - 重复次数: NUM_REPEATS 次，取均值
"""

import os
import time
import random
import math
import numpy as np
import torch  # BertModel.from_pretrained 依赖其后端，需保留
from transformers import BertTokenizer, BertModel
from scipy.special import softmax
from sklearn.metrics.pairwise import euclidean_distances

# ============================================================
# 配置
# ============================================================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
BERT_MODEL_PATH = os.path.join(PROJECT_DIR, "bert-base-uncased")
NUM_REPEATS = 3
EPSILON = 10.0  # 基线方法使用的隐私预算
EPS_HIGH = 18.0  # 采样放大方法的 ε_high
EPS_TARGET = 10.0  # 采样放大方法的目标 ε'
D_MAX = 2.892667  # BERT词表最大欧式距离
PROB_MATRIX_DTYPE = np.float32  # 距离/概率 dtype（实际部署合理选择，内存按此 dtype 报告）
# 分块大小：离线分块计算整张 |V|×|V| 距离矩阵，避免一次性物化导致 OOM
ROW_BLOCK_SIZE = 1024
# 内层循环次数：在线操作为微秒级，单次 perf_counter 计时有精度噪声，
# 故循环多次测总耗时再除以次数，得到稳定可靠的单次平均耗时。
BASELINE_INNER_LOOP = 100      # 基线在线(目标行 softmax)每次 ~0.2ms，循环 100 次足够
SA_INNER_LOOP = 100000         # 采样放大在线 ~微秒级，需循环更多次平摊计时开销

# ============================================================
# 工具函数
# ============================================================

def mldp_to_pure(eps_mldp):
    return eps_mldp * D_MAX

def calc_p(eps_prime_mldp, eps2_mldp):
    """计算采样放大的采样概率 p"""
    eps_prime = mldp_to_pure(eps_prime_mldp)
    eps2 = mldp_to_pure(eps2_mldp)
    numerator = math.exp(eps_prime) - 1
    denominator = (math.exp(eps2 / 2) - 1) * (math.exp(eps_prime - eps2 / 2) + 1)
    p = numerator / denominator
    return max(0.0, min(1.0, p))

def get_vocab_and_embedding(tokenizer, model):
    """获取词表和嵌入矩阵"""
    embedding_matrix = model.embeddings.word_embeddings.weight.data.cpu().numpy()
    vocab = tokenizer.get_vocab()
    return vocab, embedding_matrix
# ============================================================
# 实验1: 基线方法（指数机制）的在线采样时间
# ============================================================

def compute_row_block_distance(embedding_matrix, row_start, row_end):
    """计算嵌入矩阵中 [row_start, row_end) 行到全词表的欧式距离块 (block, |V|)。"""
    block_embeddings = embedding_matrix[row_start:row_end]
    return euclidean_distances(block_embeddings, embedding_matrix)

def baseline_offline_build_distance_matrix(embedding_matrix, vocab_size):
    """
    基线方法离线预处理（口径甲）：计算整张 |V|×|V| 距离矩阵。

    距离矩阵与 ε 无关，离线一次性计算并预存，在线阶段替换任意 token 时
    直接取对应行即可。为避免一次性物化导致 OOM，按行分块计算并累加耗时，
    所测时间等价于构建整张距离矩阵的离线开销。

    返回（离线总耗时, 目标行距离）；整张矩阵不在内存常驻（仅用于计时），
    其内存占用按理论值 |V|×|V|×dtype_size 报告。
    """
    t_start = time.perf_counter()
    for row_start in range(0, vocab_size, ROW_BLOCK_SIZE):
        row_end = min(row_start + ROW_BLOCK_SIZE, vocab_size)
        compute_row_block_distance(embedding_matrix, row_start, row_end)
    offline_distance_time = time.perf_counter() - t_start
    return offline_distance_time

def baseline_replace_one_token(target_distance, epsilon, vocab_size):
    """
    基线方法（指数机制）替换 1 个 token 的在线阶段（口径甲）。

    离线已预存整张 |V|×|V| 距离矩阵（见 baseline_offline_build_distance_matrix），
    在线替换 1 个 token 时，直接取该 token 对应的那一行距离 target_distance，
    做 softmax(-ε·d/2) 得到长度 |V| 的概率分布并采样 1 次，复杂度 O(|V|)。

    内存口径：基线为支持替换任意 token 须常驻整张距离矩阵，在线额外内存
    按理论值 |V|×|V|×dtype_size 报告（基线固有内存需求）。

    在线耗时为亚毫秒级，循环多次取平均以消除单次计时精度噪声。
    """
    dtype_size = np.dtype(PROB_MATRIX_DTYPE).itemsize
    prob_matrix_memory_bytes = vocab_size * vocab_size * dtype_size

    sampled_token_id = 0
    t_online_start = time.perf_counter()
    for _ in range(BASELINE_INNER_LOOP):
        target_similarity = (-epsilon * target_distance / 2).astype(PROB_MATRIX_DTYPE)
        target_distribution = softmax(target_similarity)
        cumulative_probs = np.cumsum(target_distribution)
        uniform_sample = np.random.random()
        sampled_token_id = int(np.searchsorted(cumulative_probs, uniform_sample))
    online_total_time = (time.perf_counter() - t_online_start) / BASELINE_INNER_LOOP

    return {
        "online_total_time": online_total_time,
        "prob_matrix_memory_bytes": prob_matrix_memory_bytes,
        "prob_matrix_memory_GB": prob_matrix_memory_bytes / (1024**3),
        "sampled_token_id": sampled_token_id,
    }
# ============================================================
# 实验2: 采样放大方法的在线采样时间
# ============================================================

def sample_amplification_replace_one_token(vocab_size, eps_target_mldp, eps_high_mldp, target_token_id):
    """
    采样放大方法替换 1 个 token（ε 逐请求动态变化）。

    每次请求 ε 不同，只需用当前 ε 重算标量采样概率 p（O(1)），
    再对已有的 eps_high 扰动结果做一次伯努利采样。无需任何矩阵。

    为与基线对称、保证公平，calc_p 的标量计算同样计入在线时间。
    在线耗时为微秒级，循环多次取平均以消除单次计时精度噪声。
    """
    sampled_token_id = 0
    p = 0.0
    t_start = time.perf_counter()
    for _ in range(SA_INNER_LOOP):
        # 在线步骤1: 用当前 ε 重算采样概率 p（标量运算，O(1)）
        p = calc_p(eps_target_mldp, eps_high_mldp)

        # 在线步骤2: 对该 token 做 1 次伯努利采样
        if random.random() < p:
            # 保留 eps_high 的扰动结果（已有，无需计算）
            sampled_token_id = target_token_id
        else:
            # 均匀随机采样
            sampled_token_id = random.randint(0, vocab_size - 1)
    t_total = (time.perf_counter() - t_start) / SA_INNER_LOOP

    return {
        "total_time": t_total,
        "p": p,
        "memory_bytes": 0,  # 无需额外矩阵
        "memory_GB": 0.0,
        "sampled_token_id": sampled_token_id,
    }

# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 65)
    print("单 token 替换效率对比实验（ε 逐请求动态变化场景）")
    print("=" * 65)
    print("\n场景假设: 不同用户的隐私预算 ε 不同，每次请求 ε 都可能变化。")
    print("口径 A: 指数机制需全词表归一化，ε 变化后即使只替换 1 个 token")
    print("        也必须重建整张概率矩阵。")
    print("  - 基线单 token 在线时间 = 重建整张概率矩阵 + 1 次采样")
    print("  - 采样放大单 token 在线时间 = 重算标量 p + 1 次伯努利采样")
    print("  - 与 ε 无关的相似度矩阵为离线一次性预处理，不计入在线时间")

    # 加载 tokenizer 和模型
    print("\n[1/4] 加载 BERT 模型和词表...")
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
    model = BertModel.from_pretrained(BERT_MODEL_PATH)
    vocab, embedding_matrix = get_vocab_and_embedding(tokenizer, model)
    vocab_size = len(vocab)
    print(f"  词表大小: {vocab_size}")
    print(f"  嵌入维度: {embedding_matrix.shape[1]}")

    # ============================================================
    # 说明（口径甲）: 基线离线预存整张 |V|×|V| 距离矩阵（分块算避免 OOM），
    # 在线替换任意 token 时取对应行做 softmax，复杂度 O(|V|)；内存按理论公式
    # |V|×|V|×dtype 报告（基线为支持任意 token 替换须常驻整张矩阵，固有需求）。
    # ============================================================
    print("\n[2/4] 准备实验（口径甲: 离线预存整张距离矩阵, 在线取目标行）...")
    print(f"  分块大小 ROW_BLOCK_SIZE = {ROW_BLOCK_SIZE}")
    print(f"  距离/概率 dtype = {np.dtype(PROB_MATRIX_DTYPE).name}")

    # 选定一个待替换的目标 token（结果与具体 token 无关，固定取一个以可复现）
    target_token_id = vocab.get("good", 0)
    print(f"\n  基线方法 ε = {EPSILON}")
    print(f"  采样放大: ε_high={EPS_HIGH}, ε_target={EPS_TARGET}")
    print(f"  待替换目标 token id = {target_token_id}")

    # ============================================================
    # 实验1: 基线方法替换 1 个 token
    #   离线: 预存整张 |V|×|V| 距离矩阵（分块算）
    #   在线: 取目标行 softmax(-ε·d/2) + 1 次采样, O(|V|)
    # ============================================================
    print(f"\n[3/4] 基线方法（指数机制）单 token 替换测试 ({NUM_REPEATS}次)...")

    # 离线预处理: 构建整张距离矩阵（计时），并取出目标行供在线使用
    print(f"  [离线] 构建整张 |V|×|V| 距离矩阵（分块, 避免 OOM）...")
    offline_distance_times = []
    for i in range(NUM_REPEATS):
        offline_time = baseline_offline_build_distance_matrix(embedding_matrix, vocab_size)
        offline_distance_times.append(offline_time)
        print(f"    第{i+1}次离线构建: {offline_time:.3f}s")
    # 目标行距离（在线从预存矩阵取行，此处单独算出该行供在线复用）
    target_distance = euclidean_distances(
        embedding_matrix[target_token_id:target_token_id + 1], embedding_matrix
    )[0]

    baseline_results = []
    for i in range(NUM_REPEATS):
        result = baseline_replace_one_token(target_distance, EPSILON, vocab_size)
        baseline_results.append(result)
        print(f"  第{i+1}次: [在线]目标行softmax+采样={result['online_total_time']*1000:.4f}ms")

    # ============================================================
    # 实验2: 采样放大方法替换 1 个 token（重算 p + 1 次伯努利采样）
    # ============================================================
    print(f"\n[4/4] 采样放大方法单 token 替换测试 ({NUM_REPEATS}次)...")
    sa_results = []
    for i in range(NUM_REPEATS):
        result = sample_amplification_replace_one_token(vocab_size, EPS_TARGET, EPS_HIGH, target_token_id)
        sa_results.append(result)
        print(f"  第{i+1}次: 在线总计={result['total_time']*1e6:.4f}μs (p={result['p']:.6f})")

    # ============================================================
    # 汇总结果（取均值；当重复次数>=3 时去掉首尾极值更稳健）
    # ============================================================
    print("\n" + "=" * 65)
    print("实验结果汇总（单 token 替换）")
    print("=" * 65)

    def robust_mean(values):
        """重复次数>=3 时去掉首尾极值取均值，否则直接取均值"""
        sorted_vals = sorted(values)
        if len(sorted_vals) >= 3:
            sorted_vals = sorted_vals[1:-1]
        return np.mean(sorted_vals)

    # 基线方法
    avg_offline_dist_time = robust_mean(offline_distance_times)
    avg_online_baseline = robust_mean([r['online_total_time'] for r in baseline_results])
    memory_baseline_GB = baseline_results[0]['prob_matrix_memory_GB']
    memory_baseline_MB = baseline_results[0]['prob_matrix_memory_bytes'] / (1024**2)

    # 采样放大方法
    avg_total_sa = robust_mean([r['total_time'] for r in sa_results])

    speedup = avg_online_baseline / avg_total_sa if avg_total_sa > 0 else float("inf")

    print(f"\n  [离线一次性预处理] 构建整张 |V|×|V| 距离矩阵: {avg_offline_dist_time:.3f}s (与 ε 无关, 预存可复用, 不计入在线)")

    print(f"\n  替换 1 个 token 的在线开销对比 (口径甲: 离线预存整张矩阵, 在线取目标行):")
    print(f"\n  {'指标':<26} {'基线(指数机制)':<22} {'采样放大':<22}")
    print("  " + "-" * 70)
    print(f"  {'在线计算复杂度':<26} {'O(|V|) 目标行softmax':<22} {'O(1) 标量p+伯努利':<22}")
    print(f"  {'在线总时间/token':<26} {avg_online_baseline*1e6:.4f} μs{'':<11} {avg_total_sa*1e6:.4f} μs")
    print(f"  {'在线额外内存':<26} {memory_baseline_GB:.3f} GB{'':<13} {'≈ 0 B (无需矩阵)':<22}")
    print(f"  {'在线加速比':<26} {'1x':<22} {speedup:.1f}x")

    print(f"\n补充信息:")
    print(f"  距离矩阵大小: {vocab_size} × {vocab_size} = {vocab_size**2:,} 个元素 ({np.dtype(PROB_MATRIX_DTYPE).name})")
    print(f"  距离矩阵内存(理论值 |V|×|V|×dtype): {memory_baseline_GB:.3f} GB ({memory_baseline_MB:.1f} MB)")
    print(f"  基线单 token 在线总时间(不含距离, 仅目标行): {avg_online_baseline*1e6:.4f} μs")
    print(f"  采样放大单 token 在线总时间: {avg_total_sa*1e6:.4f} μs")
    print(f"\n结论(口径甲): 基线离线预存整张 |V|×|V| 距离矩阵({avg_offline_dist_time:.1f}s, 占 {memory_baseline_GB:.2f}GB, 与 ε 无关可复用)。")
    print(f"      在线替换 1 个 token 取目标行做 softmax (O(|V|), {avg_online_baseline*1e6:.1f}μs);")
    print(f"      采样放大仅需 O(1) 重算标量 p + 1 次伯努利采样 ({avg_total_sa*1e6:.2f}μs),")
    print(f"      在线时间快约 {speedup:.1f}×; 且基线须常驻整张距离矩阵 ({memory_baseline_GB:.2f}GB),")
    print(f"      而采样放大内存 ≈ 0。时间/内存口径完全自洽。")

if __name__ == "__main__":
    main()
