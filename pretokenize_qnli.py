"""
对原始 QNLI 数据进行 BERT WordPiece 预分词，输出格式与 SanText 脱敏后的数据一致。

目的：
  SanText 脱敏流程中，文本在脱敏前已经过 BertTokenizer.tokenize()，
  输出的 TSV 中存储的是空格拼接的 subword token。
  而 run_glue.py 中的 processor_glue.py 对输入做 .split(" ") 后直接
  调用 convert_tokens_to_ids()，不会再次 tokenize。

  因此，如果直接用原始文本跑 run_glue.py，大量词汇（大写、未分词）
  会被映射为 [UNK]，导致不公平对比。

  本脚本对原始数据做同样的 BertTokenizer.tokenize() 预分词，
  存储到新目录，保证与脱敏数据在同等条件下进入微调。

输入：  data/QNLI/train.tsv, data/QNLI/dev.tsv
输出：  output_QNLI_bert_pretokenized/train.tsv, output_QNLI_bert_pretokenized/dev.tsv

格式：  index\tquestion\tsentence\tlabel  （与脱敏后数据格式一致）

用法：
  python3 pretokenize_qnli.py
  python3 pretokenize_qnli.py --bert_model_path ./bert-base-uncased --data_dir ./data/QNLI --output_dir ./output_QNLI_bert_pretokenized
"""

import os
import argparse
from tqdm import tqdm
from transformers import BertTokenizer


def main():
    parser = argparse.ArgumentParser(description="对原始 QNLI 数据进行 BERT 预分词")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/QNLI",
        help="原始 QNLI 数据目录（包含 train.tsv 和 dev.tsv）",
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
        default="./output_QNLI_bert_pretokenized",
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
                if len(parts) < 4:
                    # 格式异常，原样写入
                    wf.write(line + "\n")
                    continue

                idx = parts[0]
                question = parts[1]
                sentence = parts[2]
                label = parts[3]

                # 使用 BertTokenizer.tokenize() 进行 WordPiece 分词
                # 这与 run_SanText.py 中的处理方式完全一致
                question_tokens = tokenizer.tokenize(question)
                sentence_tokens = tokenizer.tokenize(sentence)

                # 用空格拼接 subword tokens（与脱敏数据格式一致）
                question_tokenized = " ".join(question_tokens)
                sentence_tokenized = " ".join(sentence_tokens)

                wf.write(f"{idx}\t{question_tokenized}\t{sentence_tokenized}\t{label}\n")

        print(f"完成: {output_path}")

    # 验证：打印前几行对比
    print("\n" + "=" * 70)
    print("验证：原始数据 vs 预分词数据（前 2 条）")
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

        for i in range(1, min(3, len(orig_lines))):
            orig_parts = orig_lines[i].rstrip("\n").split("\t")
            tok_parts = tok_lines[i].rstrip("\n").split("\t")

            print(f"\n--- 第 {i} 条 ---")
            print(f"  原始 question:   {orig_parts[1][:80]}")
            print(f"  分词 question:   {tok_parts[1][:80]}")
            print(f"  原始 sentence:   {orig_parts[2][:80]}")
            print(f"  分词 sentence:   {tok_parts[2][:80]}")

    print(f"\n预分词数据已保存至: {args.output_dir}")
    print("原始数据未被修改。")


if __name__ == "__main__":
    main()
