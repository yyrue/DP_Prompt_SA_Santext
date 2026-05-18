"""
热力图实验：固定 eps_target，用不同 (eps_low, eps_high) 组合混合

原理：
  已有多份不同 epsilon 的脱敏文档（eps = 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24），
  选取任意两份作为 eps_low 和 eps_high，通过逐 token 混合采样，
  生成满足 eps_target 的脱敏文档。

通用采样概率公式（Section 3.3）：
  p = (e^{ε'} - e^{ε_low}) / [(e^{ε_high} - e^{ε'}) * e^{-(ε_high + ε_low)/2} + (e^{ε'} - e^{ε_low})]

  其中：
    ε_high: 高质量文档的 pure-DP epsilon
    ε_low:  低质量文档的 pure-DP epsilon
    ε':     目标 pure-DP epsilon (eps_target * d_max)

  约束：eps_low < eps_target < eps_high

输出目录结构：
  output_SST2_bert_utility_heatmap_target{eps_target}/
    low{eps_low}_high{eps_high}/run_1/train.tsv
    low{eps_low}_high{eps_high}/run_1/dev.tsv
    ...
"""

import os
import math
import random
import argparse
import itertools
from tqdm import tqdm

# -------------------------------------------------------
# 常量
# -------------------------------------------------------
D_MAX = 2.892667          # BERT word embedding 最大欧式距离
RUNS = [1, 2, 3, 4, 5]
FILES = ["train.tsv", "dev.tsv"]

# 可用的 epsilon 值（对应 output_SST2_bert_utility 下的目录）
AVAILABLE_EPSILONS = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]

BASE_DIR = "/data/youyaru/youyaru/SanText-main/output_SST2_bert_utility"


def mldp_to_pure(eps_mldp: float) -> float:
    """MLDP epsilon 转换为 pure-DP epsilon"""
    return eps_mldp * D_MAX


def calc_p_general(eps_low_mldp: float, eps_high_mldp: float, eps_target_mldp: float) -> float:
    """
    通用混合采样概率公式（论文 Lemma 3.3）

    p = (e^{ε'} - e^{ε₁}) / [(e^{(ε₁+ε₂)/2} - e^{ε₁}) + (1 - e^{(ε₁-ε₂)/2}) * e^{ε'}]

    其中：
        ε₁ = eps_low  (低质量文档的 pure-DP epsilon)
        ε₂ = eps_high (高质量文档的 pure-DP epsilon)
        ε' = eps_target (目标 pure-DP epsilon)
        约束: ε₁ < ε' < ε₂

    参数：
        eps_low_mldp:    低质量文档的 MLDP epsilon (ε₁)
        eps_high_mldp:   高质量文档的 MLDP epsilon (ε₂)
        eps_target_mldp: 目标 MLDP epsilon (ε')
    返回：
        p: 选择高质量文档 token 的概率
    """
    # 转换为 pure-DP epsilon
    eps1 = mldp_to_pure(eps_low_mldp)
    eps2 = mldp_to_pure(eps_high_mldp)
    eps_prime = mldp_to_pure(eps_target_mldp)

    e1 = math.exp(eps1)
    ep = math.exp(eps_prime)

    numerator = ep - e1
    denominator = (math.exp((eps1 + eps2) / 2) - e1) + (1 - math.exp((eps1 - eps2) / 2)) * ep

    if denominator == 0:
        return 0.0

    p = numerator / denominator

    # 数值保护：p 应在 [0, 1] 范围内
    p = max(0.0, min(1.0, p))
    return p


def read_tsv(filepath: str):
    """
    读取 tsv 文件，自动检测是否有 header
    返回 (header_or_None, [(text, label), ...])
    """
    header = None
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if not lines:
        return None, []

    # 检测第一行是否为 header
    first_parts = lines[0].strip().split("\t")
    if first_parts[-1] == "label" or first_parts[0] == "sentence":
        header = lines[0]
        lines = lines[1:]

    for line in lines:
        line = line.rstrip("\n")
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) >= 2:
            text = "\t".join(parts[:-1])
            label = parts[-1]
        else:
            text = parts[0]
            label = ""
        rows.append((text, label))

    return header, rows


def mix_documents(rows_low, rows_high, p: float, seed: int):
    """
    对两份文档逐 token 混合采样

    对每个句子的每个 token：
      - 以概率 p 选择 rows_high（高质量）的 token
      - 以概率 1-p 选择 rows_low（低质量）的 token
    """
    rng = random.Random(seed)
    mixed_rows = []

    min_doc_len = min(len(rows_low), len(rows_high))
    rows_low = rows_low[:min_doc_len]
    rows_high = rows_high[:min_doc_len]

    for (text_low, label_low), (text_high, label_high) in zip(rows_low, rows_high):
        tokens_low = text_low.split()
        tokens_high = text_high.split()

        min_len = min(len(tokens_low), len(tokens_high))
        max_len = max(len(tokens_low), len(tokens_high))

        mixed_tokens = []
        for i in range(max_len):
            if i < min_len:
                if rng.random() < p:
                    mixed_tokens.append(tokens_high[i])
                else:
                    mixed_tokens.append(tokens_low[i])
            elif i < len(tokens_high):
                if rng.random() < p:
                    mixed_tokens.append(tokens_high[i])
            else:
                if rng.random() >= p:
                    mixed_tokens.append(tokens_low[i])

        mixed_text = " ".join(mixed_tokens)
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
    parser = argparse.ArgumentParser(
        description="热力图实验：固定 eps_target，用不同 (eps_low, eps_high) 组合混合"
    )
    parser.add_argument("--eps_target", type=float, default=10.0,
                        help="目标 MLDP epsilon（默认 10.0）")
    parser.add_argument("--eps_low_list", type=float, nargs="+", default=None,
                        help="eps_low 列表（默认自动选取所有 < eps_target 的可用值）")
    parser.add_argument("--eps_high_list", type=float, nargs="+", default=None,
                        help="eps_high 列表（默认自动选取所有 > eps_target 的可用值）")
    parser.add_argument("--runs", type=int, nargs="+", default=RUNS,
                        help="run 编号列表（默认 1-5）")
    parser.add_argument("--d_max", type=float, default=D_MAX,
                        help=f"BERT embedding 最大欧式距离（默认 {D_MAX}）")
    parser.add_argument("--base_dir", type=str, default=BASE_DIR,
                        help="原始脱敏数据根目录")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出根目录（默认自动生成）")
    args = parser.parse_args()

    # 自动确定 eps_low_list 和 eps_high_list
    if args.eps_low_list is None:
        args.eps_low_list = [e for e in AVAILABLE_EPSILONS if e < args.eps_target]
    if args.eps_high_list is None:
        args.eps_high_list = [e for e in AVAILABLE_EPSILONS if e > args.eps_target]

    # 输出目录
    if args.output_dir is None:
        args.output_dir = os.path.join(
            os.path.dirname(args.base_dir),
            f"output_SST2_bert_utility_heatmap_target{int(args.eps_target)}"
        )

    # 生成所有 (eps_low, eps_high) 组合
    combinations = list(itertools.product(args.eps_low_list, args.eps_high_list))

    print("=" * 70)
    print("热力图实验：混合采样脱敏文本生成")
    print("=" * 70)
    print(f"  d_max          = {args.d_max}")
    print(f"  eps_target     = {args.eps_target}")
    print(f"  eps_low 列表   = {args.eps_low_list}")
    print(f"  eps_high 列表  = {args.eps_high_list}")
    print(f"  组合数         = {len(combinations)}")
    print(f"  runs           = {args.runs}")
    print(f"  输出目录       = {args.output_dir}")
    print()

    # 预先打印所有组合的 p 值
    print(f"{'eps_low':>8}  {'eps_high':>9}  {'p':>12}  {'状态'}")
    print("-" * 50)
    valid_combinations = []
    for eps_low, eps_high in combinations:
        p = calc_p_general(eps_low, eps_high, args.eps_target)
        status = "OK" if 0 < p < 1 else "SKIP (p out of range)"
        print(f"{eps_low:>8.1f}  {eps_high:>9.1f}  {p:>12.8f}  {status}")
        if 0 < p < 1:
            valid_combinations.append((eps_low, eps_high, p))

    print(f"\n有效组合数: {len(valid_combinations)} / {len(combinations)}")
    print()

    if not valid_combinations:
        print("没有有效的组合，退出。")
        return

    # -------------------------------------------------------
    # 主循环：遍历所有有效组合 × run × file
    # -------------------------------------------------------
    total = len(valid_combinations) * len(args.runs) * len(FILES)
    pbar = tqdm(total=total, desc="生成进度")

    for eps_low, eps_high, p in valid_combinations:
        combo_name = f"low{eps_low:.0f}_high{eps_high:.0f}"

        for run in args.runs:
            # 读取 eps_low 和 eps_high 的文件
            for fname in FILES:
                path_low = os.path.join(
                    args.base_dir,
                    f"eps_{eps_low:.2f}", f"run_{run}",
                    f"eps_{eps_low:.2f}", fname
                )
                path_high = os.path.join(
                    args.base_dir,
                    f"eps_{eps_high:.2f}", f"run_{run}",
                    f"eps_{eps_high:.2f}", fname
                )

                # 检查文件是否存在
                if not os.path.exists(path_low):
                    print(f"\n[WARN] 文件不存在: {path_low}，跳过")
                    pbar.update(1)
                    continue
                if not os.path.exists(path_high):
                    print(f"\n[WARN] 文件不存在: {path_high}，跳过")
                    pbar.update(1)
                    continue

                # 读取数据
                header_low, rows_low = read_tsv(path_low)
                header_high, rows_high = read_tsv(path_high)

                # 混合采样（种子包含 eps_low, eps_high, run 信息以保证唯一性）
                seed = int(eps_low * 100 + eps_high * 10 + run)
                mixed_rows = mix_documents(rows_low, rows_high, p, seed)

                # 输出路径
                out_path = os.path.join(
                    args.output_dir,
                    combo_name, f"run_{run}", fname
                )

                # 使用 header（如果有的话）
                header = header_high if header_high else header_low
                if header is None:
                    header = "sentence\tlabel\n"
                write_tsv(out_path, header, mixed_rows)
                pbar.update(1)

    pbar.close()

    # -------------------------------------------------------
    # 输出汇总信息
    # -------------------------------------------------------
    print()
    print("=" * 70)
    print("完成！")
    print("=" * 70)
    print(f"\n输出目录: {args.output_dir}")
    print(f"\n有效组合及采样概率:")
    print(f"{'eps_low':>8}  {'eps_high':>9}  {'p':>12}")
    print("-" * 35)
    for eps_low, eps_high, p in valid_combinations:
        print(f"{eps_low:>8.1f}  {eps_high:>9.1f}  {p:>12.8f}")

    # 保存组合信息到 CSV（供后续绘图使用）
    info_path = os.path.join(args.output_dir, "combinations_info.csv")
    with open(info_path, "w") as f:
        f.write("eps_low,eps_high,eps_target,p\n")
        for eps_low, eps_high, p in valid_combinations:
            f.write(f"{eps_low},{eps_high},{args.eps_target},{p}\n")
    print(f"\n组合信息已保存: {info_path}")


if __name__ == "__main__":
    main()
