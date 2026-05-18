#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
复现 SanText 论文 Figure 4
需要:
  1. 准确率数据 (来自 experiment_results_SST2.csv)
  2. 防御率数据 (需要运行 bert_inference_token.py 或手动计算)
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse
import torch
import random
from tqdm import tqdm
from scipy.special import softmax
from functools import partial
from multiprocessing import Pool, cpu_count
from sklearn.metrics.pairwise import euclidean_distances
from spacy.lang.en import English
from transformers import BertTokenizer, BertForMaskedLM
import copy
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def cal_probability(word_embed_1, word_embed_2, epsilon=2.0):
    distance = euclidean_distances(word_embed_1, word_embed_2)
    sim_matrix = -distance
    prob_matrix = softmax(epsilon * sim_matrix / 2, axis=1)
    return prob_matrix


def SanText_init(prob_matrix_init, word2id_init, sword2id_init, all_words_init, p_init, tokenizer_init):
    global prob_matrix, word2id, sword2id, id2sword, all_words, p, tokenizer
    prob_matrix = prob_matrix_init
    word2id = word2id_init
    sword2id = sword2id_init
    id2sword = {v: k for k, v in sword2id.items()}
    all_words = all_words_init
    p = p_init
    tokenizer = tokenizer_init


def SanText(doc):
    new_doc = []
    for word in doc:
        if word in word2id:
            if word in sword2id:
                index = word2id[word]
                sampling_prob = prob_matrix[index]
                sampling_index = np.random.choice(len(sampling_prob), 1, p=sampling_prob)
                new_doc.append(id2sword[sampling_index[0]])
            else:
                flip_p = random.random()
                if flip_p <= p:
                    index = word2id[word]
                    sampling_prob = prob_matrix[index]
                    sampling_index = np.random.choice(len(sampling_prob), 1, p=sampling_prob)
                    new_doc.append(id2sword[sampling_index[0]])
                else:
                    new_doc.append(word)
        else:
            sampling_prob = 1 / len(all_words) * np.ones(len(all_words), )
            sampling_index = np.random.choice(len(sampling_prob), 1, p=sampling_prob)
            new_doc.append(all_words[sampling_index[0]])
    return new_doc


def compute_defense_rate(data_dir, model_path, embedding_type='glove',
                         word_embedding_path='./data/glove.840B.300d.txt',
                         epsilon=2.0, p=0.2, sensitive_word_percentage=0.5,
                         task='SST-2', max_seq_length=64, batch_size=256,
                         threads=12, seed=42):
    """
    计算防御率
    返回: defense_rate, attack_success_rate
    """
    set_seed(seed)

    print(f"计算防御率: epsilon={epsilon}, task={task}")

    # 加载 tokenizer
    if embedding_type == "glove":
        tokenizer = English()
        tokenizer_type = "word"
    else:
        tokenizer = BertTokenizer.from_pretrained(model_path)
        tokenizer_type = "subword"

    # 构建 vocabulary
    from utils import get_vocab_SST2, get_vocab_CliniSTS, get_vocab_QNLI, word_normalize

    if task == "SST-2":
        vocab = get_vocab_SST2(data_dir, tokenizer, tokenizer_type=tokenizer_type)
    elif task == "CliniSTS":
        vocab = get_vocab_CliniSTS(data_dir, tokenizer, tokenizer_type=tokenizer_type)
    elif task == "QNLI":
        vocab = get_vocab_QNLI(data_dir, tokenizer, tokenizer_type=tokenizer_type)
    else:
        raise NotImplementedError

    # 确定敏感词
    sensitive_word_count = int(sensitive_word_percentage * len(vocab))
    words = [key for key, _ in vocab.most_common()]
    sensitive_words = words[-sensitive_word_count - 1:]
    sensitive_words2id = {word: k for k, word in enumerate(sensitive_words)}

    print(f"#Total Words: {len(words)}, #Sensitive Words: {len(sensitive_words2id)}")

    # 加载词嵌入
    word2id = {}
    sword2id = {}
    sensitive_word_embed = []
    all_word_embed = []
    sensitive_count = 0
    all_count = 0

    if embedding_type == "glove":
        num_lines = sum(1 for _ in open(word_embedding_path))
        print(f"Loading Word Embedding: {word_embedding_path}")

        with open(word_embedding_path) as f:
            line = f.readline().rstrip().split(' ')
            if len(line) != 2:
                f.seek(0)
            for row in tqdm(f, total=num_lines - 1):
                content = row.rstrip().split(' ')
                cur_word = word_normalize(content[0])
                if cur_word in vocab and cur_word not in word2id:
                    word2id[cur_word] = all_count
                    all_count += 1
                    emb = [float(i) for i in content[1:]]
                    all_word_embed.append(emb)
                    if cur_word in sensitive_words2id:
                        sword2id[cur_word] = sensitive_count
                        sensitive_count += 1
                        sensitive_word_embed.append(emb)
            f.close()
    else:
        print(f"Loading BERT Embedding: {model_path}")
        model = BertForMaskedLM.from_pretrained(model_path)
        embedding_matrix = model.bert.embeddings.word_embeddings.weight.data.cpu().numpy()

        for cur_word in tokenizer.vocab:
            if cur_word in vocab and cur_word not in word2id:
                word2id[cur_word] = all_count
                emb = embedding_matrix[tokenizer.convert_tokens_to_ids(cur_word)]
                all_word_embed.append(emb)
                all_count += 1

                if cur_word in sensitive_words2id:
                    sword2id[cur_word] = sensitive_count
                    sensitive_count += 1
                    sensitive_word_embed.append(emb)

    all_word_embed = np.array(all_word_embed, dtype='f')
    sensitive_word_embed = np.array(sensitive_word_embed, dtype='f')

    # 计算概率矩阵
    print("Calculating Prob Matrix...")
    prob_matrix = cal_probability(all_word_embed, sensitive_word_embed, epsilon)

    # 读取原始数据
    print(f"Reading data from: {data_dir}")
    original_docs = []
    labels = []

    with open(os.path.join(data_dir, 'dev.tsv'), 'r') as f:
        header = next(f)
        for line in f:
            content = line.strip().split("\t")
            if task == "SST-2":
                text = content[0]
                label = int(content[1])
                if embedding_type == "glove":
                    doc = [token.text for token in tokenizer(text)]
                else:
                    doc = tokenizer.tokenize(text)
                original_docs.append(doc)
                labels.append(label)
            elif task == "QNLI":
                text1 = content[1]
                text2 = content[2]
                label = content[-1]
                if embedding_type == "glove":
                    doc1 = [token.text for token in tokenizer(text1)]
                    doc2 = [token.text for token in tokenizer(text2)]
                else:
                    doc1 = tokenizer.tokenize(text1)
                    doc2 = tokenizer.tokenize(text2)
                original_docs.append(doc1)
                labels.append(label)

    # 对文本进行清洗
    print("Sanitizing documents...")
    threads = min(threads, cpu_count())
    with Pool(threads, initializer=SanText_init,
              initargs=(prob_matrix, word2id, sword2id, words, p, tokenizer)) as pool:
        sanitized_docs = list(tqdm(pool.imap(SanText, original_docs, chunksize=32),
                                   total=len(original_docs)))
        pool.close()

    # 加载 BERT 模型用于攻击
    print("Loading BERT for attack...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_model = BertForMaskedLM.from_pretrained(model_path)
    bert_model.to(device)
    bert_model = torch.nn.DataParallel(bert_model)
    bert_model.eval()

    # 执行 mask token inference attack
    print("Running mask token inference attack...")
    tokenized_new_docs = []
    ground_truth_labels = []

    for i, new_doc in enumerate(sanitized_docs):
        assert len(original_docs[i]) == len(new_doc)
        for j in range(len(new_doc)):
            tmp_doc = copy.deepcopy(new_doc)
            tmp_doc[j] = " MASK]"
            tokenized_new_docs.append(tokenizer.encode_plus(
                tmp_doc, padding="max_length", max_length=max_seq_length, truncation=True))
            ground_truth_labels.append(tokenizer.convert_tokens_to_ids(original_docs[i][j]))

    all_input_ids = torch.tensor([doc.data['input_ids'] for doc in tokenized_new_docs], dtype=torch.long)
    all_token_type_ids = torch.tensor([doc.data['token_type_ids'] for doc in tokenized_new_docs], dtype=torch.long)
    all_attention_mask = torch.tensor([doc.data['attention_mask'] for doc in tokenized_new_docs], dtype=torch.long)
    all_labels = torch.tensor(ground_truth_labels, dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_token_type_ids, all_attention_mask, all_labels)
    dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=batch_size)

    intersect_num = 0
    total_num = 0

    for batch in tqdm(dataloader):
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[2],
                "token_type_ids": batch[1]
            }
            prediction = bert_model(**inputs)[0]
            prediction = torch.argmax(prediction, dim=2)
            prediction = prediction[torch.where(batch[0] == 103)]
            ground_truths = batch[3]
            intersect_num += (prediction == ground_truths).sum().item()
            total_num += len(prediction)

    attack_success_rate = intersect_num / total_num
    defense_rate = 1.0 - attack_success_rate

    print(f"Attack Success Rate: {attack_success_rate:.4f}")
    print(f"Defense Rate: {defense_rate:.4f}")

    return defense_rate, attack_success_rate


def load_accuracy_data(csv_path):
    """从 CSV 文件加载准确率数据"""
    data = defaultdict(list)
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epsilon = float(row['epsilon'])
            accuracy = float(row['accuracy'])
            data[epsilon].append(accuracy)

    # 计算每个 epsilon 的平均值
    result = {}
    for eps, accs in data.items():
        result[eps] = np.mean(accs)

    return result


def plot_figure4(accuracy_data, defense_data, output_path='figure4.png', dataset_name='SST-2'):
    """
    绘制 Figure 4

    accuracy_data: {epsilon: accuracy}
    defense_data: {epsilon: defense_rate}
    """
    # 准备数据
    epsilons = sorted(accuracy_data.keys())
    accuracies = [accuracy_data[eps] for eps in epsilons]
    defense_rates = [defense_data.get(eps, 0) for eps in epsilons]

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 7))

    # 绘制曲线
    ax.plot(accuracies, defense_rates, 'bo-', linewidth=2, markersize=10, label='SanText')

    # 添加 epsilon 标注
    for i, eps in enumerate(epsilons):
        ax.annotate(f'ε={eps}', (accuracies[i], defense_rates[i]),
                   textcoords="offset points", xytext=(10, 5), fontsize=10)

    # 添加未清洗的点 (epsilon = infinity)
    # 假设未清洗时: 准确率约 0.92, 防御率约 0.4 (根据论文)
    if len(epsilons) > 0:
        ax.axhline(y=0.4, color='r', linestyle='--', alpha=0.5, label='Unsanitized (ε=∞)')
        ax.axvline(x=0.92, color='r', linestyle='--', alpha=0.5)

    # 设置标签
    ax.set_xlabel('Task Utility (Accuracy)', fontsize=14)
    ax.set_ylabel('Defense Rate (Privacy)', fontsize=14)
    ax.set_title(f'Privacy and Utility Tradeoffs on {dataset_name}', fontsize=16)

    # 设置范围
    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(0.0, 1.0)

    # 添加网格
    ax.grid(True, alpha=0.3)

    # 添加图例
    ax.legend(loc='best', fontsize=12)

    # 保存图片
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {output_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='复现 SanText 论文 Figure 4')

    parser.add_argument('--task', type=str, default='SST-2', choices=['SST-2', 'QNLI', 'CliniSTS'])
    parser.add_argument('--data_dir', type=str, default='./data/SST-2/')
    parser.add_argument('--model_path', type=str, default='bert-base-uncased')
    parser.add_argument('--embedding_type', type=str, default='glove', choices=['glove', 'bert'])
    parser.add_argument('--word_embedding_path', type=str, default='./data/glove.840B.300d.txt')
    parser.add_argument('--accuracy_csv', type=str, default='./experiment_results_SST2.csv')
    parser.add_argument('--output_path', type=str, default='./figure4_SST2.png')

    # epsilon 列表
    parser.add_argument('--epsilon_list', type=float, nargs='+', default=[1.0, 2.0, 3.0])

    # 其他参数
    parser.add_argument('--p', type=float, default=0.2)
    parser.add_argument('--sensitive_word_percentage', type=float, default=0.5)
    parser.add_argument('--max_seq_length', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--threads', type=int, default=12)
    parser.add_argument('--seed', type=int, default=42)

    # 模式选择
    parser.add_argument('--mode', type=str, default='full', choices=['full', 'plot_only', 'defense_only'])
    parser.add_argument('--defense_results', type=str, default='./defense_results.csv',
                       help='保存/读取防御率结果的文件')

    args = parser.parse_args()

    if args.mode == 'plot_only':
        # 只绘图，从文件读取防御率数据
        defense_data = {}
        if os.path.exists(args.defense_results):
            with open(args.defense_results, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    defense_data[float(row['epsilon'])] = float(row['defense_rate'])

        accuracy_data = load_accuracy_data(args.accuracy_csv)
        plot_figure4(accuracy_data, defense_data, args.output_path, args.task)

    elif args.mode == 'defense_only':
        # 只计算防御率
        defense_data = {}
        for eps in args.epsilon_list:
            defense_rate, _ = compute_defense_rate(
                args.data_dir, args.model_path, args.embedding_type,
                args.word_embedding_path, eps, args.p, args.sensitive_word_percentage,
                args.task, args.max_seq_length, args.batch_size, args.threads, args.seed
            )
            defense_data[eps] = defense_rate

        # 保存结果
        with open(args.defense_results, 'w') as f:
            f.write('epsilon,defense_rate,attack_success_rate\n')
            for eps, dr in defense_data.items():
                f.write(f'{eps},{dr},{1-dr}\n')
        print(f"Defense results saved to: {args.defense_results}")

    else:  # full mode
        # 完整流程
        # 1. 加载准确率
        accuracy_data = load_accuracy_data(args.accuracy_csv)
        print(f"Loaded accuracy data: {accuracy_data}")

        # 2. 计算防御率
        defense_data = {}
        for eps in args.epsilon_list:
            defense_rate, _ = compute_defense_rate(
                args.data_dir, args.model_path, args.embedding_type,
                args.word_embedding_path, eps, args.p, args.sensitive_word_percentage,
                args.task, args.max_seq_length, args.batch_size, args.threads, args.seed
            )
            defense_data[eps] = defense_rate

        # 保存防御率结果
        with open(args.defense_results, 'w') as f:
            f.write('epsilon,defense_rate\n')
            for eps, dr in defense_data.items():
                f.write(f'{eps},{dr}\n')

        # 3. 绘图
        plot_figure4(accuracy_data, defense_data, args.output_path, args.task)


if __name__ == '__main__':
    main()