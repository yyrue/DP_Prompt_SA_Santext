#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算 GloVe 词向量中任意两个词向量之间的最大欧氏距离

策略：
  GloVe 840B 共 ~220万词，全量 N×N 距离矩阵约需 220万² × 4 bytes ≈ 不可行
  采用"分块扫描"方法：
    1. 一次性加载全部词向量（~220万 × 300 维，约 2.5GB float32）
    2. 分块计算距离矩阵，只保留每块的最大值，不存储完整矩阵
    3. 最终取所有块最大值的最大值
  若内存不足，可通过 --vocab_limit 限制词表大小
"""

import numpy as np
import argparse
import time
from tqdm import tqdm

# ============================================================
# 参数
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument('--glove_path', type=str,
                    default='data/glove.840B.300d.txt',
                    help='GloVe 文件路径')
parser.add_argument('--vocab_limit', type=int, default=0,
                    help='限制加载词数（0=全部加载）')
parser.add_argument('--block_size', type=int, default=5000,
                    help='分块大小（每次计算 block_size × N 的距离）')
args = parser.parse_args()

# ============================================================
# 1. 加载词向量
# ============================================================
print(f"[1/3] 加载 GloVe 词向量: {args.glove_path}")
print(f"      vocab_limit = {'全部' if args.vocab_limit == 0 else args.vocab_limit}")

embeddings = []
words = []
t0 = time.time()

with open(args.glove_path, 'r', encoding='utf-8', errors='ignore') as f:
    # 检测是否有 "词数 维度" 的头行（word2vec 格式）
    first_line = f.readline().rstrip().split(' ')
    if len(first_line) == 2:
        print(f"      检测到头行: {first_line}，已跳过")
    else:
        f.seek(0)

    for i, row in enumerate(tqdm(f, desc='读取词向量', unit='词')):
        if args.vocab_limit > 0 and i >= args.vocab_limit:
            break
        parts = row.rstrip().split(' ')
        if len(parts) < 10:          # 跳过异常行
            continue
        try:
            emb = np.array(parts[1:], dtype=np.float32)
            if emb.shape[0] != 300:  # 维度校验
                continue
            embeddings.append(emb)
            words.append(parts[0])
        except ValueError:
            continue

embeddings = np.array(embeddings, dtype=np.float32)   # shape: (N, 300)
N = len(embeddings)
print(f"      加载完成: {N} 个词向量，维度 {embeddings.shape[1]}")
print(f"      耗时: {time.time()-t0:.1f}s，内存占用约 {embeddings.nbytes/1e9:.2f} GB")

# ============================================================
# 2. 分块计算最大欧氏距离
# ============================================================
print(f"\n[2/3] 分块计算最大欧氏距离 (block_size={args.block_size})...")
print(f"      共 {N} 个词，需要 {(N + args.block_size - 1) // args.block_size} 个块")

# 欧氏距离公式：||a - b||² = ||a||² + ||b||² - 2·a·b^T
# 预计算每个向量的 L2 范数平方
norms_sq = np.sum(embeddings ** 2, axis=1)   # shape: (N,)

global_max_dist = 0.0
global_max_pair = (0, 0)

t1 = time.time()
block_size = args.block_size
n_blocks = (N + block_size - 1) // block_size

for i in tqdm(range(n_blocks), desc='分块计算', unit='块'):
    i_start = i * block_size
    i_end   = min(i_start + block_size, N)
    block_i = embeddings[i_start:i_end]          # (B, 300)
    norms_i = norms_sq[i_start:i_end]            # (B,)

    # 只计算上三角（j >= i），避免重复
    for j in range(i, n_blocks):
        j_start = j * block_size
        j_end   = min(j_start + block_size, N)
        block_j = embeddings[j_start:j_end]      # (B', 300)
        norms_j = norms_sq[j_start:j_end]        # (B',)

        # 距离平方矩阵: (B, B')
        dot = block_i @ block_j.T                # (B, B')
        dist_sq = (norms_i[:, None]
                   + norms_j[None, :]
                   - 2.0 * dot)
        # 数值误差修正（避免负数开根）
        dist_sq = np.maximum(dist_sq, 0.0)

        # 若是对角块，排除自身（对角线为0）
        if i == j:
            np.fill_diagonal(dist_sq, 0.0)

        local_max_sq = dist_sq.max()
        if local_max_sq > global_max_dist ** 2:
            global_max_dist = float(np.sqrt(local_max_sq))
            idx = np.unravel_index(np.argmax(dist_sq), dist_sq.shape)
            global_max_pair = (i_start + idx[0], j_start + idx[1])

print(f"      分块计算耗时: {time.time()-t1:.1f}s")

# ============================================================
# 3. 输出结果
# ============================================================
w1, w2 = words[global_max_pair[0]], words[global_max_pair[1]]
v1, v2 = embeddings[global_max_pair[0]], embeddings[global_max_pair[1]]
verify_dist = float(np.linalg.norm(v1 - v2))

print("\n" + "=" * 55)
print("  GloVe 840B 最大欧氏距离计算结果")
print("=" * 55)
print(f"  词表大小       : {N:,} 个词")
print(f"  最大欧氏距离   : {global_max_dist:.6f}")
print(f"  验证距离       : {verify_dist:.6f}")
print(f"  最远词对       : '{w1}'  ↔  '{w2}'")
print(f"  词对索引       : ({global_max_pair[0]}, {global_max_pair[1]})")
print("=" * 55)
print(f"\n  总耗时: {time.time()-t0:.1f}s")
