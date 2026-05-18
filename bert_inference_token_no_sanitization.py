"""
Mask Token Inference Attack - 无脱敏版本 (Baseline)

此脚本直接对原始数据进行Mask Token Inference Attack
不使用SanText进行脱敏，用于与脱敏后的结果进行对比

流程：
1. 加载原始数据
2. 对每个token进行mask
3. 使用BERT MLM预测被mask的token
4. 计算defense rate（预测失败的比例）
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


def set_seed(args):
    """设置随机种子"""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def main():
    parser = argparse.ArgumentParser()
    
    # 模型参数
    parser.add_argument(
        "--model_path",
        default="bert-base-uncased",
        type=str,
        help="BERT模型路径"
    )
    
    # 数据参数
    parser.add_argument(
        "--data_dir",
        default="./data/SST-2/",
        type=str,
        help="数据目录"
    )
    
    parser.add_argument(
        "--output_dir",
        default="./output_no_sanitization/",
        type=str,
        help="输出目录"
    )
    
    parser.add_argument(
        "--max_seq_length",
        default=64,
        type=int,
        help="最大序列长度"
    )
    
    parser.add_argument(
        "--batch_size",
        default=256,
        type=int,
        help="批大小"
    )
    
    parser.add_argument(
        '--task',
        choices=['CliniSTS', "SST-2", "QNLI"],
        default='SST-2',
        help='任务类型'
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子"
    )
    
    args = parser.parse_args()
    
    set_seed(args)
    
    # 日志设置
    logging.basicConfig(
        format="%(asctime)s -  %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    logger.info("=" * 60)
    logger.info("Mask Token Inference Attack - 无脱敏Baseline")
    logger.info(f"任务: {args.task}")
    logger.info(f"随机种子: {args.seed}")
    logger.info(f"模型: {args.model_path}")
    logger.info("=" * 60)
    
    # 加载BERT tokenizer和模型
    logger.info("加载BERT模型和tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    model = BertForMaskedLM.from_pretrained(args.model_path)
    
    # 读取原始数据
    logger.info("读取原始数据...")
    docs = []
    labels = []
    
    for file_name in ['dev.tsv']:
        data_file = os.path.join(args.data_dir, file_name)
        logger.info(f"处理文件: {data_file}")
        
        num_lines = sum(1 for _ in open(data_file))
        with open(data_file, 'r') as rf:
            # 跳过header
            header = next(rf)
            
            if args.task == "SST-2":
                for line in tqdm(rf, total=num_lines - 1, desc="读取数据"):
                    content = line.strip().split("\t")
                    text = content[0]
                    label = int(content[1])
                    # 使用BERT tokenizer分词
                    doc = tokenizer.tokenize(text)
                    docs.append(doc)
                    labels.append(label)
            
            elif args.task == "QNLI":
                for line in tqdm(rf, total=num_lines - 1, desc="读取数据"):
                    content = line.strip().split("\t")
                    text1 = content[1]
                    text2 = content[2]
                    label = content[-1]
                    doc1 = tokenizer.tokenize(text1)
                    doc2 = tokenizer.tokenize(text2)
                    docs.append(doc1)
                    # docs.append(doc2)  # 可选：是否包含第二个句子
                    labels.append(label)
            
            rf.close()
    
    logger.info(f"共读取 {len(docs)} 条数据")
    
    # 统计token数量
    total_tokens = sum(len(doc) for doc in docs)
    logger.info(f"总token数: {total_tokens}")
    
    # 构建攻击数据
    # 对原始数据中的每个token进行mask
    logger.info("构建攻击数据...")
    
    tokenized_docs = []
    ground_truth_labels = []
    
    for i, doc in enumerate(tqdm(docs, desc="构建mask数据")):
        for j in range(len(doc)):
            # 创建一个副本，将第j个token替换为[MASK]
            tmp_doc = copy.deepcopy(doc)
            tmp_doc[j] = "[MASK]"
            
            # tokenize
            tokenized_input = tokenizer.encode_plus(
                tmp_doc,
                padding="max_length",
                max_length=args.max_seq_length,
                truncation=True
            )
            tokenized_docs.append(tokenized_input)
            
            # ground truth是原始token的id
            ground_truth_labels.append(tokenizer.convert_tokens_to_ids(doc[j]))
    
    # 转换为tensor
    all_input_ids = torch.tensor(
        [doc.data['input_ids'] for doc in tokenized_docs],
        dtype=torch.long
    )
    all_token_type_ids = torch.tensor(
        [doc.data['token_type_ids'] for doc in tokenized_docs],
        dtype=torch.long
    )
    all_attention_mask = torch.tensor(
        [doc.data['attention_mask'] for doc in tokenized_docs],
        dtype=torch.long
    )
    all_labels = torch.tensor(ground_truth_labels, dtype=torch.long)
    
    # 创建数据集和数据加载器
    dataset = TensorDataset(
        all_input_ids,
        all_token_type_ids,
        all_attention_mask,
        all_labels
    )
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size)
    
    # 移动模型到GPU
    if args.max_seq_length <= 0:
        args.max_seq_length = tokenizer.max_len_single_sentence
    args.max_seq_length = min(args.max_seq_length, tokenizer.max_len_single_sentence)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    model.to(device)
    
    # 如果有多个GPU，使用DataParallel
    if torch.cuda.device_count() > 1:
        logger.info(f"使用 {torch.cuda.device_count()} 个GPU")
        model = torch.nn.DataParallel(model)
    
    # 进行预测
    logger.info("开始预测...")
    
    intersect_num = 0  # 预测正确的数量
    total_num = 0       # 总预测数量
    
    for batch in tqdm(dataloader, desc="预测中"):
        batch = tuple(t.to(device) for t in batch)
        
        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[2],  # 注意：batch[1]是token_type_ids
                "token_type_ids": batch[1]
            }
            
            # BERT预测
            prediction = model(**inputs)[0]  # [batch_size, seq_len, vocab_size]
            
            # 取最大概率的token
            prediction = torch.argmax(prediction, dim=2)
            
            # 找到[MASK]位置（token id = 103）
            mask_positions = torch.where(batch[0] == 103)
            
            # 提取[MASK]位置的预测
            prediction_at_mask = prediction[mask_positions]
            
            # ground truth
            ground_truths = batch[3]
            
            # 统计预测正确的数量
            intersect_num += (prediction_at_mask == ground_truths).sum().item()
            total_num += len(prediction_at_mask)
    
    # 计算结果
    attack_success_rate = intersect_num / total_num
    defense_rate = 1.0 - attack_success_rate
    
    # 打印结果
    logger.info("=" * 60)
    logger.info("结果:")
    logger.info(f"  预测正确数: {intersect_num}")
    logger.info(f"  总预测数: {total_num}")
    logger.info(f"  攻击成功率 (Attack Success Rate): {attack_success_rate:.4f} ({attack_success_rate*100:.2f}%)")
    logger.info(f"  防御率 (Defense Rate): {defense_rate:.4f} ({defense_rate*100:.2f}%)")
    logger.info("=" * 60)
    
    # 保存结果到CSV
    results_file = os.path.join(os.path.dirname(args.output_dir), "mask_attack_results_no_sanitization.csv")
    
    # 如果文件不存在，写入header
    if not os.path.exists(results_file):
        with open(results_file, 'w') as f:
            f.write("run,seed,defense_rate,attack_success_rate\n")
    
    # 从output_dir提取run编号
    run_num = args.output_dir.split('_')[-1] if '_' in args.output_dir else "1"
    
    # 追加结果
    with open(results_file, 'a') as f:
        f.write(f"{run_num},{args.seed},{defense_rate:.4f},{attack_success_rate}\n")
    
    logger.info(f"结果已保存到: {results_file}")
    
    # 同时打印标准输出格式（方便脚本捕获）
    print(intersect_num)
    print(total_num)
    print(defense_rate)


if __name__ == "__main__":
    main()