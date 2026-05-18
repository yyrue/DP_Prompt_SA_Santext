"""
Mask Token Inference Attack - 直接读取已脱敏数据版本

此脚本直接读取已经脱敏好的数据（如 mixed 数据），
不再进行脱敏步骤，直接对脱敏后的 token 进行 Mask Token Inference Attack。

攻击流程：
1. 读取原始 dev.tsv（获取 ground truth token）
2. 读取脱敏后的 dev.tsv（获取 sanitized token）
3. 对脱敏后的每个 token 位置做 [MASK]
4. 用 BERT MLM 预测被 mask 的位置
5. 与原始 token 比较，计算攻击成功率
"""

import argparse
import torch
import random
import numpy as np
import logging
import os
import copy
from tqdm import tqdm
from transformers import BertTokenizer, BertForMaskedLM
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler

logger = logging.getLogger(__name__)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_sanitized_tsv(filepath, has_header=True):
    """读取脱敏后的 tsv 文件，返回 token 列表的列表"""
    docs = []
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    if has_header:
        lines = lines[1:]
    for line in lines:
        line = line.rstrip('\n')
        if not line:
            continue
        parts = line.split('\t')
        text = '\t'.join(parts[:-1])   # 最后一列是 label
        tokens = text.split()          # 脱敏数据已经是 subword token，空格分隔
        docs.append(tokens)
    return docs


def read_original_tsv(filepath, tokenizer, has_header=True):
    """读取原始 tsv 文件，用 BERT tokenizer 分词，返回 token 列表的列表"""
    docs = []
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    if has_header:
        lines = lines[1:]
    for line in lines:
        line = line.rstrip('\n')
        if not line:
            continue
        parts = line.split('\t')
        text = parts[0]
        tokens = tokenizer.tokenize(text)
        docs.append(tokens)
    return docs


def main():
    parser = argparse.ArgumentParser(
        description="Mask Token Inference Attack on pre-sanitized data"
    )

    parser.add_argument("--model_path", default="bert-base-uncased", type=str,
                        help="BERT 模型路径")
    parser.add_argument("--original_data_dir", default="./data/SST-2/", type=str,
                        help="原始数据目录（用于获取 ground truth token）")
    parser.add_argument("--sanitized_data_dir", type=str, required=True,
                        help="已脱敏数据目录（直接读取，不再脱敏）")
    parser.add_argument("--output_dir", default="./output_attack/", type=str,
                        help="攻击结果输出目录")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="最大序列长度")
    parser.add_argument("--batch_size", default=256, type=int,
                        help="批大小")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")

    args = parser.parse_args()
    set_seed(args.seed)

    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info("=" * 60)
    logger.info("Mask Token Inference Attack（直接读取脱敏数据）")
    logger.info(f"原始数据:   {args.original_data_dir}")
    logger.info(f"脱敏数据:   {args.sanitized_data_dir}")
    logger.info(f"随机种子:   {args.seed}")
    logger.info("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)

    # -------------------------------------------------------
    # 加载 BERT tokenizer 和模型
    # -------------------------------------------------------
    logger.info("加载 BERT 模型...")
    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    model = BertForMaskedLM.from_pretrained(args.model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    model.to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # -------------------------------------------------------
    # 读取原始 dev.tsv（ground truth）
    # -------------------------------------------------------
    original_dev = os.path.join(args.original_data_dir, "dev.tsv")
    logger.info(f"读取原始数据: {original_dev}")
    original_docs = read_original_tsv(original_dev, tokenizer, has_header=True)
    logger.info(f"原始数据条数: {len(original_docs)}")

    # -------------------------------------------------------
    # 读取脱敏后的 dev.tsv（sanitized tokens）
    # -------------------------------------------------------
    sanitized_dev = os.path.join(args.sanitized_data_dir, "dev.tsv")
    logger.info(f"读取脱敏数据: {sanitized_dev}")

    # 判断脱敏文件是否有 header
    with open(sanitized_dev, 'r') as f:
        first_line = f.readline()
    has_header = ('sentence' in first_line or 'label' in first_line)
    sanitized_docs = read_sanitized_tsv(sanitized_dev, has_header=has_header)
    logger.info(f"脱敏数据条数: {len(sanitized_docs)}")

    # -------------------------------------------------------
    # 对齐原始和脱敏数据（行数可能因 header 差异略有不同）
    # -------------------------------------------------------
    n = min(len(original_docs), len(sanitized_docs))
    original_docs  = original_docs[:n]
    sanitized_docs = sanitized_docs[:n]
    logger.info(f"对齐后数据条数: {n}")

    # -------------------------------------------------------
    # 构建攻击数据集
    # 对脱敏后的每个 token 位置做 [MASK]，ground truth 是原始 token
    # -------------------------------------------------------
    logger.info("构建攻击数据集...")
    tokenized_inputs = []
    ground_truth_ids = []
    skipped = 0

    for i in tqdm(range(n), desc="构建 mask 数据"):
        orig_tokens = original_docs[i]
        san_tokens  = sanitized_docs[i]

        # 以较短的为准（两者 token 数可能不同）
        min_len = min(len(orig_tokens), len(san_tokens))

        for j in range(min_len):
            # 脱敏后的句子，将第 j 个 token 替换为 [MASK]
            tmp_doc = copy.deepcopy(san_tokens)
            tmp_doc[j] = "[MASK]"

            encoded = tokenizer.encode_plus(
                tmp_doc,
                padding="max_length",
                max_length=args.max_seq_length,
                truncation=True
            )
            tokenized_inputs.append(encoded)

            # ground truth：原始 token 对应的 id
            gt_id = tokenizer.convert_tokens_to_ids(orig_tokens[j])
            ground_truth_ids.append(gt_id)

    logger.info(f"总攻击样本数（token 级别）: {len(tokenized_inputs)}")

    # -------------------------------------------------------
    # 转换为 Tensor，构建 DataLoader
    # 顺序与原始 bert_inference_token.py 完全一致：
    #   batch[0] = input_ids
    #   batch[1] = attention_mask   ← 注意：原始代码 batch[1] 传给 attention_mask
    #   batch[2] = token_type_ids   ← 原始代码 batch[2] 传给 token_type_ids
    #   batch[3] = labels（原始 token id）
    # -------------------------------------------------------
    all_input_ids      = torch.tensor([x.data['input_ids']      for x in tokenized_inputs], dtype=torch.long)
    all_attention_mask = torch.tensor([x.data['attention_mask'] for x in tokenized_inputs], dtype=torch.long)
    all_token_type_ids = torch.tensor([x.data['token_type_ids'] for x in tokenized_inputs], dtype=torch.long)
    all_labels         = torch.tensor(ground_truth_ids, dtype=torch.long)

    # TensorDataset 顺序：(input_ids, attention_mask, token_type_ids, labels)
    dataset    = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    sampler    = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size)

    # -------------------------------------------------------
    # 推理攻击
    # -------------------------------------------------------
    logger.info("开始 Mask Token Inference Attack...")
    intersect_num = 0
    total_num     = 0

    for batch in tqdm(dataloader, desc="攻击推理"):
        batch = tuple(t.to(device) for t in batch)
        # batch[0]: input_ids
        # batch[1]: attention_mask
        # batch[2]: token_type_ids
        # batch[3]: labels（原始 token id）
        with torch.no_grad():
            inputs = {
                "input_ids":      batch[0],
                "attention_mask": batch[1],   # 与原始 bert_inference_token.py 保持一致
                "token_type_ids": batch[2],
            }
            logits = model(**inputs)[0]                    # (B, seq_len, vocab_size)
            pred   = torch.argmax(logits, dim=2)           # (B, seq_len)

            # 找 [MASK] 位置（token id = 103）
            # 每个样本构建时只有一个 [MASK]，所以 mask_positions[0] 即行索引
            mask_positions = torch.where(batch[0] == 103)
            pred_at_mask   = pred[mask_positions]          # (num_masks,)

            # ground truth：每行对应一个原始 token id，用行索引对齐
            gt = batch[3][mask_positions[0]]               # (num_masks,)

            intersect_num += (pred_at_mask == gt).sum().item()
            total_num     += len(pred_at_mask)

    # -------------------------------------------------------
    # 输出结果
    # -------------------------------------------------------
    attack_success_rate = intersect_num / total_num if total_num > 0 else 0.0
    defense_rate        = 1.0 - attack_success_rate

    logger.info("=" * 60)
    logger.info("攻击结果:")
    logger.info(f"  预测正确数:    {intersect_num}")
    logger.info(f"  总预测数:      {total_num}")
    logger.info(f"  攻击成功率:    {attack_success_rate:.4f} ({attack_success_rate*100:.2f}%)")
    logger.info(f"  防御率:        {defense_rate:.4f} ({defense_rate*100:.2f}%)")
    logger.info("=" * 60)

    # 标准输出（供 bash 脚本捕获）
    print(intersect_num)
    print(total_num)
    print(attack_success_rate)


if __name__ == "__main__":
    main()
