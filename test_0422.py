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


def load_dev_data(data_dir):
    dev_path = os.path.join(data_dir,"dev.tsv")
    docs = []
    with open(dev_path,"r",encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines[1:]:
        parts = line.strip().split("\t") #strip()去除换行符，split("\t")按制表符分割
        if len(parts) >= 4:
            question = parts[1]
            sentence = parts[2]
            text = question + " " + sentence
            docs.append(text.split())
    return docs
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
BERT_MODEL_PATH = os.path.join(PROJECT_DIR,"bert-base-uncased")

tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)

#加载模型
model = BertModel.from_pretrained(BERT_MODEL_PATH)

vocab = tokenizer.get_vocab()
print(type(vocab)) #字典



# print(vocab.keys())
# print(vocab.values())

# words = list(vocab.keys())
# print(words)
# print(vocab["ter"])


# id = tokenizer.encode("ter",add_special_tokens=False)#编码，false不添加特殊token
# print(id)

# print(vocab)
print(list(vocab.keys())[-1])
print(vocab)

token_id = tokenizer.convert_tokens_to_ids("[PAD]")
print(token_id)
embedding_matrix = model.embeddings.word_embeddings.weight.data.cpu().numpy()
v1 = embedding_matrix[token_id]
print("嵌入矩阵：",v1)

