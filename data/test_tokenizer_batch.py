from transformers import AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("/data/youyaru/youyaru/SanText-main/bert-base-uncased")

raw_samples = [
    {"text":"I love NLP","label":0},
    {"text":"Transformers are powerful","label":1},
    {"text":"BERT uses wordPiece tokenization.","label":1},
    {"text":"Hi!","label":0},
]

batch_size = 2
max_length = 16

# input_ids = torch.full((batch_size,max_length),fill_value=tokenizer.pad_token_id,dtype=torch.long)
# print(input_ids.shape)

# def pad_to_max_length(batch_input_ids, pad_token_id,max_length):
#     # batch_input_idsL list of list[int]
#     batch_size = len(batch_input_ids)
#     input_ids = torch.full((batch_size,max_length),pad_token_id,dtype=torch.long)
#     attention_mask = torch.zeros((batch_size,max_length),dtype=torch.long)
#     for i,ids in enumerate(batch_input_ids):
#         length = min(len(ids),max_length)
#         #做padding
#         input_ids[i,:length] = torch.tensor(ids[:length],dtype=torch.long)
#         attention_mask[i,:length] = 1
#     return input_ids, attention_mask


# pad_id = tokenizer.pad_token_id
# print(pad_id) #就是vocab中的第一个

# for start in range(0,len(raw_samples),batch_size):
#     batch = raw_samples[start:start + batch_size]
#     # 准备：每条样本分别 tokenizer
#     tokenized = []
#     labels = []
#     for sample in batch:
#         text = sample["text"]
#         labels.append(sample["label"])
#         out = tokenizer(
#             text,
#             add_special_tokens = True,
#             truncation = True,
#             max_length = max_length,
#             padding = False,
#         )
#         print(out)
#         tokenized.append(out["input_ids"])

# input_ids, attention_mask = pad_to_max_length(tokenized,pad_id,max_length)
# labels = torch.tensor(labels,dtype=torch.long)
# print(batch)
# print("===新的batch===")
# print("batch文本：",[s["text"] for s in batch])
# print("tokenized input_ids:",tokenized)
# print("input_ids padded shape:",input_ids.shape)
# print("attention_mask shape:",attention_mask.shape)
# print("labels shape:",labels.shape)



for start in range(0,len(raw_samples),batch_size):
    batch = raw_samples[start:start + batch_size]
    texts = [s["text"] for s in batch]
    labels = torch.tensor([s["label"] for s in batch],dtype=torch.long)

    enc = tokenizer(
        texts,
        add_special_tokens = True,
        truncation = True,
        max_length = max_length,
        padding = "max_length",
        return_tensors = "pt"
    )
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    print("=======新的batch========")
    print("text:",texts)
    for i in range(len(input_ids)):
        print("input_ids:",input_ids[i])
    print("input_ids shape:",input_ids.shape)
    print("attention_mask:",attention_mask)