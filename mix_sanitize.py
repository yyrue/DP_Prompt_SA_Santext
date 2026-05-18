"""
基于混合采样生成中间 epsilon' 的脱敏文本

原理：
  已有 eps_0（完全随机）和 eps_2（高质量脱敏）两份文档，
  对每个 token 以概率 p 选择 eps_2 的 token，以概率 (1-p) 选择 eps_0 的 token，
  混合后的文档满足 epsilon' 的 MLDP 隐私保证。

采样概率公式（SanText+ 论文）：
  p = (e^{ε'} - 1) / [(e^{ε/2} - 1) * (e^{ε' - ε/2} + 1)]

  其中：
    ε  = ε_2（高质量文档的 MLDP epsilon，这里是 24）
    ε' = 目标 MLDP epsilon（2, 4, 6, ..., 22）
    ε_1 = 0（随机文档对应的 epsilon）

  注意：公式中的 epsilon 需要先转换为 pure-DP epsilon：
    ε_pure = ε_MLDP * d_max   (d_max = 2.892667，BERT word embedding 最大欧式距离)

输出目录结构：
  output_SST2_bert_utility_mixed/
    eps_2.00/run_1/eps_2.00/train.tsv
    eps_2.00/run_1/eps_2.00/dev.tsv
    ...
"""

import os
import math
import random
import argparse
from tqdm import tqdm

# -------------------------------------------------------
# 常量
# -------------------------------------------------------
D_MAX = 2.892667          # BERT word embedding 最大欧式距离
EPS_HIGH = 24.0           # 高质量文档的 MLDP epsilon（ε_2），可自行指定
EPS_LOW = 0.0             # 随机文档的 MLDP epsilon（ε_1 = 0）
# TARGET_EPSILONS 根据 EPS_HIGH 自动生成：[2, 4, 6, ..., EPS_HIGH-2]
#TARGET_EPSILONS = list(range(2, int(EPS_HIGH), 2))
TARGET_EPSILONS = [0]
RUNS = [1, 2, 3, 4, 5]
FILES = ["train.tsv", "dev.tsv"]

BASE_DIR = "/home/youyaru/SanText-main/output_SST2_bert_utility"
OUTPUT_BASE_DIR = f"/home/youyaru/SanText-main/output_SST2_bert_utility_mixed_eps{int(EPS_HIGH)}"


def mldp_to_pure(eps_mldp: float) -> float:
    """MLDP epsilon 转换为 pure-DP epsilon"""
    return eps_mldp * D_MAX


def calc_p(eps_prime_mldp: float, eps2_mldp: float) -> float:
    """
    计算混合采样概率 p

    公式：p = (e^{ε'} - 1) / [(e^{ε/2} - 1) * (e^{ε' - ε/2} + 1)]

    其中 ε', ε 均为 pure-DP epsilon（已乘以 d_max）
    ε_1 = 0（随机文档），所以公式已化简为上述形式

    参数：
        eps_prime_mldp: 目标 MLDP epsilon
        eps2_mldp:      高质量文档的 MLDP epsilon
    返回：
        p: 选择高质量文档 token 的概率
    """
    # 转换为 pure-DP epsilon
    eps_prime = mldp_to_pure(eps_prime_mldp)
    eps2     = mldp_to_pure(eps2_mldp)

    # 公式计算
    numerator   = math.exp(eps_prime) - 1
    denominator = (math.exp(eps2 / 2) - 1) * (math.exp(eps_prime - eps2 / 2) + 1)

    p = numerator / denominator

    # 数值保护：p 应在 [0, 1] 范围内
    p = max(0.0, min(1.0, p))
    return p


def read_tsv(filepath: str, has_header: bool):
    """
    读取 tsv 文件，返回 (header_or_None, [(text, label), ...])

    eps_0.00 文件没有 header，eps_24.00 有 header
    """
    header = None
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if has_header:
        header = lines[0]
        lines = lines[1:]

    for line in lines:
        line = line.rstrip("\n")
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) >= 2:
            text  = "\t".join(parts[:-1])
            label = parts[-1]
        else:
            text  = parts[0]
            label = ""
        rows.append((text, label))

    return header, rows


def mix_documents(rows_low, rows_high, p: float, seed: int):
    """
    对两份文档逐 token 混合采样

    对每个句子的每个 token：
      - 以概率 p 选择 rows_high（高质量）的 token
      - 以概率 1-p 选择 rows_low（随机）的 token

    参数：
        rows_low:  [(text, label), ...]  eps=0 文档
        rows_high: [(text, label), ...]  eps=24 文档
        p:         选择高质量 token 的概率
        seed:      随机种子
    返回：
        mixed_rows: [(text, label), ...]
    """
    rng = random.Random(seed)
    mixed_rows = []

    assert len(rows_low) == len(rows_high), \
        f"行数不一致: low={len(rows_low)}, high={len(rows_high)}"

    for (text_low, label_low), (text_high, label_high) in zip(rows_low, rows_high):
        tokens_low  = text_low.split()
        tokens_high = text_high.split()

        # 两份文档 token 数可能不同（因为随机替换可能改变 subword 数量）
        # 以较短的为准，逐 token 采样
        min_len = min(len(tokens_low), len(tokens_high))
        max_len = max(len(tokens_low), len(tokens_high))

        mixed_tokens = []
        for i in range(max_len):
            if i < min_len:
                # 两边都有 token，按概率 p 选择
                if rng.random() < p:
                    mixed_tokens.append(tokens_high[i])
                else:
                    mixed_tokens.append(tokens_low[i])
            elif i < len(tokens_high):
                # 只有 high 有，直接用 high（或按 p 决定是否保留）
                if rng.random() < p:
                    mixed_tokens.append(tokens_high[i])
            else:
                # 只有 low 有，直接用 low（或按 1-p 决定是否保留）
                if rng.random() >= p:
                    mixed_tokens.append(tokens_low[i])

        mixed_text = " ".join(mixed_tokens)
        # label 以 high 文档为准（两者应相同）
        mixed_rows.append((mixed_text, label_high))

    return mixed_rows


def write_tsv(filepath: str, header, rows):
    """写出 tsv 文件"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        if header is not None:
            f.write(header)
        for text, label in rows:
            f.write(f"{text}\t{label}\n")


def main():
    parser = argparse.ArgumentParser(description="混合采样生成中间 epsilon 脱敏文本")
    parser.add_argument("--eps_high", type=float, default=EPS_HIGH,
                        help=f"高质量文档的 MLDP epsilon（默认 {EPS_HIGH}）")
    parser.add_argument("--target_epsilons", type=float, nargs="+",
                        default=TARGET_EPSILONS if TARGET_EPSILONS else None,
                        help="目标 MLDP epsilon 列表（默认使用全局变量 TARGET_EPSILONS，若为空则自动生成 [2,4,...,eps_high-2]）")
    parser.add_argument("--runs", type=int, nargs="+", default=RUNS,
                        help="run 编号列表")
    parser.add_argument("--d_max", type=float, default=D_MAX,
                        help=f"BERT embedding 最大欧式距离（默认 {D_MAX}）")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出根目录（默认根据 eps_high 自动生成）")
    args = parser.parse_args()

    # 如果未指定 target_epsilons，根据 eps_high 自动生成 [2, 4, 6, ..., eps_high-2]
    if args.target_epsilons is None:
        args.target_epsilons = list(range(2, int(args.eps_high), 2))

    # 如果未指定 output_dir，根据 eps_high 自动生成
    if args.output_dir is None:
        args.output_dir = f"/home/youyaru/SanText-main/output_SST2_bert_utility_mixed_eps{int(args.eps_high)}"

    print("=" * 65)
    print("混合采样脱敏文本生成")
    print("=" * 65)
    print(f"  d_max          = {args.d_max}")
    print(f"  ε_high (MLDP)  = {args.eps_high}  →  pure-DP = {mldp_to_pure(args.eps_high):.4f}")
    print(f"  ε_low  (MLDP)  = {EPS_LOW}   →  pure-DP = {mldp_to_pure(EPS_LOW):.4f}")
    print(f"  目标 ε' 列表   = {args.target_epsilons}")
    print(f"  runs           = {args.runs}")
    print(f"  输出目录       = {args.output_dir}")
    print()

    # 预先打印所有 p 值
    print(f"{'ε_MLDP':>8}  {'ε_pure':>10}  {'p':>10}")
    print("-" * 35)
    for eps_prime in sorted(args.target_epsilons):
        p = calc_p(eps_prime, args.eps_high)
        print(f"{eps_prime:>8.2f}  {mldp_to_pure(eps_prime):>10.4f}  {p:>10.6f}")
    print()

    # -------------------------------------------------------
    # 主循环：遍历所有 run × target_epsilon × file
    # -------------------------------------------------------
    total = len(args.runs) * len(args.target_epsilons) * len(FILES)
    pbar = tqdm(total=total, desc="生成进度")

    for run in args.runs:
        # 读取 eps_0 和 eps_high 的文件（每个 run 只读一次）
        data_low  = {}
        data_high = {}

        for fname in FILES:
            path_low = os.path.join(
                BASE_DIR,
                f"eps_{EPS_LOW:.2f}", f"run_{run}",
                f"eps_{EPS_LOW:.2f}", fname
            )
            path_high = os.path.join(
                BASE_DIR,
                f"eps_{args.eps_high:.2f}", f"run_{run}",
                f"eps_{args.eps_high:.2f}", fname
            )

            # eps_0 没有 header，eps_high 有 header
            # 注意：eps_0 文件第一行就是数据（无 sentence\tlabel 表头）
            #       eps_high 文件第一行是 "sentence\tlabel" 表头
            header_low,  rows_low  = read_tsv(path_low,  has_header=False)
            header_high, rows_high = read_tsv(path_high, has_header=True)

            # 两份文件数据行数应相同（eps_0 全是数据行，eps_high 去掉 header 后）
            if len(rows_low) != len(rows_high):
                # eps_0 可能也有 header（"sentence\tlabel"），尝试跳过
                if rows_low[0][1] == "label":
                    header_low, rows_low = read_tsv(path_low, has_header=True)
                # 若仍不一致，截断到较短的
                min_len = min(len(rows_low), len(rows_high))
                rows_low  = rows_low[:min_len]
                rows_high = rows_high[:min_len]

            data_low[fname]  = (header_low,  rows_low)
            data_high[fname] = (header_high, rows_high)

        # 对每个目标 epsilon 生成混合文档
        for eps_prime in sorted(args.target_epsilons):
            p = calc_p(eps_prime, args.eps_high)
            # 用 run 编号作为随机种子偏移，保证不同 run 结果不同
            seed = run * 1000 + int(eps_prime * 10)

            for fname in FILES:
                header_low,  rows_low  = data_low[fname]
                header_high, rows_high = data_high[fname]

                # 混合采样
                mixed_rows = mix_documents(rows_low, rows_high, p, seed)

                # 输出路径：与原始目录结构一致
                out_path = os.path.join(
                    args.output_dir,
                    f"eps_{eps_prime:.2f}", f"run_{run}",
                    f"eps_{eps_prime:.2f}", fname
                )

                # 输出文件使用 eps_high 的 header（有 sentence\tlabel 行）
                write_tsv(out_path, header_high, mixed_rows)
                pbar.update(1)

    pbar.close()

    print()
    print("=" * 65)
    print("完成！输出目录结构：")
    print("=" * 65)
    os.system(f"find {args.output_dir} -name '*.tsv' | sort | head -40")
    print()

    # 验证：打印每个 epsilon 的第一个 train.tsv 前 2 行
    print("=" * 65)
    print("验证：各 epsilon 的 train.tsv 前 2 行")
    print("=" * 65)
    for eps_prime in sorted(args.target_epsilons):
        path = os.path.join(
            args.output_dir,
            f"eps_{eps_prime:.2f}", "run_1",
            f"eps_{eps_prime:.2f}", "train.tsv"
        )
        if os.path.exists(path):
            with open(path) as f:
                lines = f.readlines()
            print(f"\n[ε'={eps_prime:.2f}  p={calc_p(eps_prime, args.eps_high):.4f}]  {path}")
            for line in lines[1:3]:  # 跳过 header
                print(f"  {line.rstrip()[:100]}")


if __name__ == "__main__":
    main()
