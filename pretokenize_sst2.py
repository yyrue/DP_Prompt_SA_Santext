"""
对原始 SST-2 数据进行 BERT WordPiece 预分词，输出格式与 SanText 脱敏后的数据一致。

目的：
  与 pretokenize_qnli.py 相同的原因——
  run_glue.py 中的 processor_glue.py 对输入做 .split(" ") 后直接
  调用 convert_tokens_to_ids()，不会再次 tokenize。
  如果直接用原始文本跑 run_glue.py，大量词汇会被映射为 [UNK]，
  导致与脱敏数据的对比不公平。

SST-2 数据格式：
  header:  sentence\tlabel
  数据行:  text\tlabel  (label 为 0 或 1)

输入：  data/SST-2/train.tsv, data/SST-2/dev.tsv
输出：  output_SST2_bert_pretokenized/train.tsv, output_SST2_bert_pretokenized/dev.tsv

用法：
  python3 pretokenize_sst2.py
  python3 pretokenize_sst2.py --bert_model_path ./bert-base-uncased --data_dir ./data/SST-2 --output_dir ./output_SST2_bert_pretokenized
"""

import os
import argparse
from tqdm import tqdm
from transformers import BertTokenizer


def main():
    parser = argparse.ArgumentParser(description="对原始 SST-2 数据进行 BERT 预分词")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/SST-2",
        help="原始 SST-2 数据目录（包含 train.tsv 和 dev.tsv）",
    )
    parser.add_argument(
        "--bert_model_path",
        type=str,
        default="./bert-base-uncased",
        help="BERT 模型路径（用于加载 BertTokenizer）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output_SST2_bert_pretokenized",
        help="预分词后数据的输出目录",
    )
    args = parser.parse_args()

    # 加载 tokenizer
    print(f"加载 BertTokenizer: {args.bert_model_path}")
    tokenizer = BertTokenizer.from_pretrained(args.bert_model_path)

    os.makedirs(args.output_dir, exist_ok=True)

    for file_name in ["train.tsv", "dev.tsv"]:
        input_path = os.path.join(args.data_dir, file_name)
        output_path = os.path.join(args.output_dir, file_name)

        if not os.path.exists(input_path):
            print(f"[跳过] 文件不存在: {input_path}")
            continue

        print(f"\n处理: {input_path}")
        print(f"输出: {output_path}")

        num_lines = sum(1 for _ in open(input_path, "r", encoding="utf-8"))

        with open(input_path, "r", encoding="utf-8") as rf, \
             open(output_path, "w", encoding="utf-8") as wf:

            # 读取并写入 header
            header = rf.readline()
            wf.write(header)

            for line in tqdm(rf, total=num_lines - 1, desc=f"预分词 {file_name}"):
                line = line.rstrip("\n")
                if not line:
                    continue

                parts = line.split("\t")
                if len(parts) < 2:
                    # 格式异常，原样写入
                    wf.write(line + "\n")
                    continue

                sentence = parts[0]
                label = parts[1]

                # 使用 BertTokenizer.tokenize() 进行 WordPiece 分词
                # 这与 run_SanText.py 中的处理方式完全一致
                sentence_tokens = tokenizer.tokenize(sentence)

                # 用空格拼接 subword tokens（与脱敏数据格式一致）
                sentence_tokenized = " ".join(sentence_tokens)

                wf.write(f"{sentence_tokenized}\t{label}\n")

        print(f"完成: {output_path}")

    # 验证：打印前几行对比
    print("\n" + "=" * 70)
    print("验证：原始数据 vs 预分词数据（前 3 条）")
    print("=" * 70)

    for file_name in ["train.tsv"]:
        orig_path = os.path.join(args.data_dir, file_name)
        tok_path = os.path.join(args.output_dir, file_name)

        if not os.path.exists(tok_path):
            continue

        with open(orig_path, "r", encoding="utf-8") as f:
            orig_lines = f.readlines()
        with open(tok_path, "r", encoding="utf-8") as f:
            tok_lines = f.readlines()

        for i in range(1, min(4, len(orig_lines))):
            orig_parts = orig_lines[i].rstrip("\n").split("\t")
            tok_parts = tok_lines[i].rstrip("\n").split("\t")

            print(f"\n--- 第 {i} 条 ---")
            print(f"  原始 sentence: {orig_parts[0][:80]}")
            print(f"  分词 sentence: {tok_parts[0][:80]}")
            print(f"  label:         {orig_parts[1] if len(orig_parts) > 1 else 'N/A'}")

    print(f"\n预分词数据已保存至: {args.output_dir}")
    print("原始数据未被修改。")


if __name__ == "__main__":
    main()
