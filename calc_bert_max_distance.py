"""
计算 BERT word embedding 中任意两个向量之间的最大欧式距离

BERT-base-uncased 的 word_embeddings 是静态的 Embedding 层：
  - 词表大小：30522
  - 向量维度：768
  - 权重固定，不随输入上下文变化（与 BERT 上下文向量不同）
  - 相同 token 永远得到相同的向量

由于 30522 x 30522 的完整距离矩阵约需 ~3.5GB 内存，
这里采用分块（chunk）计算，避免 OOM。
"""

import numpy as np
import torch
from transformers import BertForMaskedLM, BertTokenizer
from tqdm import tqdm
import time

BERT_MODEL_PATH = "/home/youyaru/SanText-main/bert-base-uncased"
CHUNK_SIZE = 1000  # 每次计算 1000 个向量与全部向量的距离

def main():
    print("=" * 60)
    print("加载 BERT 模型，提取 word_embeddings 权重...")
    print("=" * 60)

    model = BertForMaskedLM.from_pretrained(BERT_MODEL_PATH)
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)

    # 提取静态 word embedding 矩阵，shape: (30522, 768)
    emb = model.bert.embeddings.word_embeddings.weight.data.cpu().numpy().astype(np.float32)
    vocab_size, dim = emb.shape
    print(f"Embedding 矩阵形状: {emb.shape}  (vocab_size={vocab_size}, dim={dim})")
    print(f"数据类型: {emb.dtype}")

    # -------------------------------------------------------
    # 利用公式：||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a·b^T
    # 分块计算，避免一次性构造 30522x30522 矩阵
    # -------------------------------------------------------
    print("\n开始分块计算最大欧式距离（chunk_size={}）...".format(CHUNK_SIZE))

    # 预计算每个向量的 L2 范数平方
    norms_sq = np.sum(emb ** 2, axis=1)  # shape: (vocab_size,)

    global_max_dist = 0.0
    global_max_i = 0
    global_max_j = 0

    start_time = time.time()
    n_chunks = (vocab_size + CHUNK_SIZE - 1) // CHUNK_SIZE

    for chunk_idx in tqdm(range(n_chunks), desc="计算进度"):
        i_start = chunk_idx * CHUNK_SIZE
        i_end = min(i_start + CHUNK_SIZE, vocab_size)

        chunk = emb[i_start:i_end]  # shape: (chunk_size, 768)

        # 距离平方矩阵: (chunk_size, vocab_size)
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b^T
        dot = chunk @ emb.T  # (chunk_size, vocab_size)
        dist_sq = norms_sq[i_start:i_end, np.newaxis] + norms_sq[np.newaxis, :] - 2 * dot

        # 数值误差可能导致极小负数，clip 到 0
        dist_sq = np.clip(dist_sq, 0, None)

        # 排除自身（对角线置 0）
        for local_i in range(i_end - i_start):
            dist_sq[local_i, i_start + local_i] = 0.0

        # 找当前 chunk 的最大值
        chunk_max_idx = np.argmax(dist_sq)
        local_i, j = np.unravel_index(chunk_max_idx, dist_sq.shape)
        chunk_max_dist = np.sqrt(dist_sq[local_i, j])

        if chunk_max_dist > global_max_dist:
            global_max_dist = chunk_max_dist
            global_max_i = i_start + local_i
            global_max_j = j

    elapsed = time.time() - start_time

    # -------------------------------------------------------
    # 输出结果
    # -------------------------------------------------------
    token_i = tokenizer.convert_ids_to_tokens(int(global_max_i))
    token_j = tokenizer.convert_ids_to_tokens(int(global_max_j))

    # 验证：直接计算最远词对的距离
    vec_i = emb[global_max_i]
    vec_j = emb[global_max_j]
    verify_dist = np.linalg.norm(vec_i - vec_j)

    print("\n" + "=" * 60)
    print("计算完成！")
    print("=" * 60)
    print(f"耗时: {elapsed:.1f} 秒")
    print(f"\n最大欧式距离 (d_max): {global_max_dist:.6f}")
    print(f"验证距离（直接计算）: {verify_dist:.6f}")
    print(f"\n最远词对:")
    print(f"  Token 1: [{global_max_i}] '{token_i}'")
    print(f"  Token 2: [{global_max_j}] '{token_j}'")

    # -------------------------------------------------------
    # 额外统计：随机采样 5000 个向量的距离分布
    # -------------------------------------------------------
    print("\n" + "=" * 60)
    print("距离分布统计（随机采样 5000 个 token）")
    print("=" * 60)
    np.random.seed(42)
    sample_idx = np.random.choice(vocab_size, 5000, replace=False)
    sample_emb = emb[sample_idx]
    sample_norms_sq = np.sum(sample_emb ** 2, axis=1)
    sample_dot = sample_emb @ sample_emb.T
    sample_dist_sq = sample_norms_sq[:, np.newaxis] + sample_norms_sq[np.newaxis, :] - 2 * sample_dot
    sample_dist_sq = np.clip(sample_dist_sq, 0, None)
    np.fill_diagonal(sample_dist_sq, 0)
    sample_dist = np.sqrt(sample_dist_sq)

    upper = sample_dist[np.triu_indices(5000, k=1)]
    print(f"  样本最大距离:  {upper.max():.6f}")
    print(f"  样本最小距离:  {upper.min():.6f}")
    print(f"  样本平均距离:  {upper.mean():.6f}")
    print(f"  样本中位距离:  {np.median(upper):.6f}")
    print(f"  样本标准差:    {upper.std():.6f}")

    # -------------------------------------------------------
    # SanText 差分隐私分析
    # -------------------------------------------------------
    print("\n" + "=" * 60)
    print("SanText 差分隐私分析 (基于 d_max)")
    print("=" * 60)
    d_max = global_max_dist
    for eps in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]:
        effective = eps * d_max
        print(f"  MLDP ε={eps:2d}  →  等效标准 DP ε = {eps} × {d_max:.4f} = {effective:.4f}")

    print("\n" + "=" * 60)
    print(f"结论: BERT word embedding 最大欧式距离 d_max = {global_max_dist:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
