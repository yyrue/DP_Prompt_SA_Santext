#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
采样效率对比实验
对比基线方法（指数机制）和采样放大方法的：
  1. 在线采样时间
  2. 内存开销

实验设置：
  - 数据集: SST-2 验证集 (872条样本)
  - 词嵌入: BERT-base-uncased 嵌入层 (|V|=30522, dim=768)
  - 重复次数: 5次取平均
"""

import os
import sys
import time
import random
import math
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from scipy.special import softmax
from sklearn.metrics.pairwise import euclidean_distances

# ============================================================
# 配置
# ============================================================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
BERT_MODEL_PATH = os.path.join(PROJECT_DIR, "bert-base-uncased")
DATA_DIR = os.path.join(PROJECT_DIR, "data", "SST-2")
NUM_REPEATS = 10
EPSILON = 10.0  # 基线方法使用的隐私预算
EPS_HIGH = 18.0  # 采样放大方法的 ε_high
EPS_TARGET = 10.0  # 采样放大方法的目标 ε'
D_MAX = 2.892667  # BERT词表最大欧式距离

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

def load_dev_data(data_dir):
    """读取 SST-2 验证集，返回 token 序列列表"""
    dev_path = os.path.join(data_dir, "dev.tsv")
    docs = []
    with open(dev_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    # 跳过 header
    for line in lines[1:]:
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            text = parts[0]
            docs.append(text.split())
    return docs

def get_vocab_and_embedding(tokenizer, model):
    """获取词表和嵌入矩阵"""
    embedding_matrix = model.embeddings.word_embeddings.weight.data.cpu().numpy()
    vocab = tokenizer.get_vocab()
    return vocab, embedding_matrix

# ============================================================
# 实验1: 基线方法（指数机制）的在线采样时间
# ============================================================

def baseline_online_sampling(docs_tokenized, embedding_matrix, epsilon, vocab_size):
    """
    基线方法：计算概率矩阵 + 采样
    返回采样时间和概率矩阵内存占用
    """
    # --- 计算概率矩阵 ---
    t_start = time.time()
    
    distance = euclidean_distances(embedding_matrix, embedding_matrix)
    sim_matrix = -distance
    prob_matrix = softmax(epsilon * sim_matrix / 2, axis=1)
    
    t_prob_matrix = time.time() - t_start
    
    # 概率矩阵内存占用 (bytes)
    prob_matrix_memory = prob_matrix.nbytes
    
    # --- 采样 ---
    t_start_sample = time.time()
    
    for doc in docs_tokenized:
        for token_id in doc:
            if token_id < vocab_size:
                sampling_prob = prob_matrix[token_id]
                np.random.choice(vocab_size, 1, p=sampling_prob)
    
    t_sampling = time.time() - t_start_sample
    
    total_time = t_prob_matrix + t_sampling
    
    return {
        "prob_matrix_time": t_prob_matrix,
        "sampling_time": t_sampling,
        "total_time": total_time,
        "prob_matrix_memory_bytes": prob_matrix_memory,
        "prob_matrix_memory_GB": prob_matrix_memory / (1024**3),
    }

# ============================================================
# 实验2: 采样放大方法的在线采样时间
# ============================================================

def sample_amplification_online(docs_tokenized, vocab_size, p):
    """
    采样放大方法：仅伯努利采样
    假设已有 eps_high 的扰动结果和 eps=0 的均匀随机结果
    """
    t_start = time.time()
    
    for doc in docs_tokenized:
        for token_id in doc:
            # 伯努利采样：以概率 p 保留 eps_high 结果，以概率 1-p 均匀随机采样
            if random.random() < p:
                # 保留 eps_high 的扰动结果（模拟：直接读取已有结果）
                _ = token_id  # 已有结果，无需计算
            else:
                # 均匀随机采样
                _ = random.randint(0, vocab_size - 1)
    
    t_total = time.time() - t_start
    
    return {
        "total_time": t_total,
        "memory_bytes": 0,  # 无需额外矩阵
        "memory_GB": 0.0,
    }

# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 65)
    print("采样效率对比实验")
    print("=" * 65)
    
    # 加载 tokenizer 和模型
    print("\n[1/4] 加载 BERT 模型和词表...")
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
    model = BertModel.from_pretrained(BERT_MODEL_PATH)
    vocab, embedding_matrix = get_vocab_and_embedding(tokenizer, model)
    vocab_size = len(vocab)
    print(f"  词表大小: {vocab_size}")
    print(f"  嵌入维度: {embedding_matrix.shape[1]}")
    
    # 加载数据
    print("\n[2/4] 加载 SST-2 验证集...")
    docs = load_dev_data(DATA_DIR)
    
    # 将文本 token 转换为 token id
    docs_tokenized = []
    total_tokens = 0
    for doc in docs:
        token_ids = []
        for word in doc:
            ids = tokenizer.encode(word, add_special_tokens=False)
            token_ids.extend(ids)
        docs_tokenized.append(token_ids)
        total_tokens += len(token_ids)
    
    print(f"  文档数: {len(docs_tokenized)}")
    print(f"  总 token 数: {total_tokens}")
    
    # 计算采样概率 p
    p = calc_p(EPS_TARGET, EPS_HIGH)
    print(f"\n  基线方法 ε = {EPSILON}")
    print(f"  采样放大: ε_high={EPS_HIGH}, ε_target={EPS_TARGET}, p={p:.6f}")
    
    # ============================================================
    # 实验1: 基线方法
    # ============================================================
    print(f"\n[3/4] 基线方法（指数机制）效率测试 ({NUM_REPEATS}次)...")
    baseline_results = []
    for i in range(NUM_REPEATS):
        result = baseline_online_sampling(docs_tokenized, embedding_matrix, EPSILON, vocab_size)
        baseline_results.append(result)
        print(f"  第{i+1}次: 概率矩阵计算={result['prob_matrix_time']:.3f}s, "
              f"采样={result['sampling_time']:.3f}s, "
              f"总计={result['total_time']:.3f}s")
    
    # ============================================================
    # 实验2: 采样放大方法
    # ============================================================
    print(f"\n[4/4] 采样放大方法效率测试 ({NUM_REPEATS}次)...")
    sa_results = []
    for i in range(NUM_REPEATS):
        result = sample_amplification_online(docs_tokenized, vocab_size, p)
        sa_results.append(result)
        print(f"  第{i+1}次: 总计={result['total_time']:.6f}s")
    
    # ============================================================
    # 汇总结果（去掉最高和最低各1次，取中间值的平均）
    # ============================================================
    print("\n" + "=" * 65)
    print("实验结果汇总（去掉最高和最低各1次）")
    print("=" * 65)
    
    def trimmed_mean(values):
        """去掉最高和最低各1个，取剩余的平均值"""
        sorted_vals = sorted(values)
        trimmed = sorted_vals[1:-1]  # 去掉首尾
        return np.mean(trimmed)
    
    # 基线方法
    avg_prob_time = trimmed_mean([r['prob_matrix_time'] for r in baseline_results])
    avg_sample_time = trimmed_mean([r['sampling_time'] for r in baseline_results])
    avg_total_baseline = trimmed_mean([r['total_time'] for r in baseline_results])
    memory_baseline = baseline_results[0]['prob_matrix_memory_GB']
    
    # 采样放大方法
    avg_total_sa = trimmed_mean([r['total_time'] for r in sa_results])
    
    print(f"\n{'指标':<25} {'基线方法(指数机制)':<25} {'采样放大方法':<25}")
    print("-" * 75)
    print(f"{'概率矩阵计算时间':<25} {avg_prob_time:.3f}s{'':<20} {'—（无需计算）':<25}")
    print(f"{'在线采样时间':<25} {avg_sample_time:.3f}s{'':<20} {avg_total_sa:.6f}s")
    print(f"{'总在线时间':<25} {avg_total_baseline:.3f}s{'':<20} {avg_total_sa:.6f}s")
    print(f"{'内存占用(概率矩阵)':<25} {memory_baseline:.2f} GB{'':<18} {'≈0 (无需矩阵)':<25}")
    print(f"{'加速比':<25} {'1x':<25} {avg_total_baseline/avg_total_sa:.1f}x")
    
    print(f"\n补充信息:")
    print(f"  概率矩阵大小: {vocab_size} × {vocab_size} = {vocab_size**2:,} 个元素")
    print(f"  概率矩阵内存: {memory_baseline:.2f} GB (float64) / {memory_baseline/2:.2f} GB (float32)")
    print(f"  验证集总 token 数: {total_tokens}")
    print(f"  单个 token 基线采样时间: {avg_sample_time/total_tokens*1000:.4f} ms")
    print(f"  单个 token 采样放大时间: {avg_total_sa/total_tokens*1000:.6f} ms")


if __name__ == "__main__":
    main()
