from pytorch_pretrained_bert.tokenization import BertTokenizer
data_path="../data/zh-CN/"
with open(data_path+"IMELogs/0-30000000.txt",encoding='utf8') as f:
    for i, line in enumerate(f):
        if i % 1000 == 0:
            print(i)
