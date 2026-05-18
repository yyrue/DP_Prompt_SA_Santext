# coding=utf-8
"""
测试 get_vocab_SST2 函数在不同 tokenizer 下的输出
对比 GloVe（整词）和 BERT（子词）两种模式的词表内容
"""

import sys
sys.path.insert(0, '/Users/yyr/dp_prompt/SanText-main')

from utils import get_vocab_SST2

DATA_DIR = './data/SST-2'
BERT_MODEL = './bert-base-uncased'

# ============================================================
# 测试样例句子（不依赖数据集，直接演示分词差异）
# ============================================================
print("=" * 60)
print("【分词效果对比】")
print("=" * 60)

test_sentences = [
    "the movie is really good",
    "unhappiness and disappointment",
    "state-of-the-art performance",
    "I'm not going there",
]

# BERT 子词分词
from transformers import BertTokenizer
bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

# spaCy 整词分词
from spacy.lang.en import English
spacy_tokenizer = English()

print(f"\n{'原句':<40} {'GloVe(整词)':<40} {'BERT(子词)'}")
print("-" * 120)
for sent in test_sentences:
    glove_tokens = [t.text for t in spacy_tokenizer(sent)]
    bert_tokens  = bert_tokenizer.tokenize(sent)
    print(f"{sent:<40} {str(glove_tokens):<40} {str(bert_tokens)}")


# ============================================================
# 测试1：GloVe 模式（整词分词）
# ============================================================
print("\n" + "=" * 60)
print("【测试1】GloVe 模式（tokenizer_type=word）")
print("=" * 60)

vocab_glove = get_vocab_SST2(DATA_DIR, spacy_tokenizer, tokenizer_type="word")

print(f"\n词表总大小: {len(vocab_glove)}")
print(f"\n出现频率最高的20个词（高频词）:")
for word, count in vocab_glove.most_common(20):
    print(f"  '{word}': {count}次")

print(f"\n出现频率最低的20个词（低频词/敏感词候选）:")
for word, count in vocab_glove.most_common()[:-21:-1]:
    print(f"  '{word}': {count}次")

print(f"\n随机抽样10个词查看:")
import random
sample_words = random.sample(list(vocab_glove.keys()), 10)
for word in sample_words:
    print(f"  '{word}': {vocab_glove[word]}次")


# ============================================================
# 测试2：BERT 模式（子词分词）
# ============================================================
print("\n" + "=" * 60)
print("【测试2】BERT 模式（tokenizer_type=subword）")
print("=" * 60)

vocab_bert = get_vocab_SST2(DATA_DIR, bert_tokenizer, tokenizer_type="subword")

print(f"\n词表总大小: {len(vocab_bert)}")
print(f"\n出现频率最高的20个词（高频词）:")
for word, count in vocab_bert.most_common(20):
    print(f"  '{word}': {count}次")

print(f"\n出现频率最低的20个词（低频词/敏感词候选）:")
for word, count in vocab_bert.most_common()[:-21:-1]:
    print(f"  '{word}': {count}次")

# 查看子词特征（## 开头的是子词）
subword_tokens = [w for w in vocab_bert.keys() if w.startswith('##')]
print(f"\nBERT 子词（## 开头）数量: {len(subword_tokens)}")
print(f"子词示例: {subword_tokens[:20]}")

# 查看特殊 token
special_tokens = [w for w in vocab_bert.keys() if w.startswith('[')]
print(f"\nBERT 特殊token: {special_tokens}")


# ============================================================
# 对比总结
# ============================================================
print("\n" + "=" * 60)
print("【对比总结】")
print("=" * 60)
print(f"{'模式':<15} {'词表大小':<15} {'说明'}")
print("-" * 60)
print(f"{'GloVe(整词)':<15} {len(vocab_glove):<15} '只含数据集出现的整词'")
print(f"{'BERT(子词)':<15} {len(vocab_bert):<15} '数据集子词 + BERT完整词表(3万)'")

# 查看同一个词在两种模式下的差异
check_words = ['good', 'bad', 'movie', 'film', 'the']
print(f"\n同一个词在两种词表中的词频对比:")
print(f"{'词':<15} {'GloVe词频':<15} {'BERT词频'}")
print("-" * 40)
for w in check_words:
    g = vocab_glove.get(w, 0)
    b = vocab_bert.get(w, 0)
    print(f"  {w:<13} {g:<15} {b}")
