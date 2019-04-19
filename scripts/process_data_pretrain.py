import json
from xml.dom.minidom import parse
import xml.dom.minidom
import os
import re
from pytorch_pretrained_bert.tokenization import BertTokenizer


tokenizer = BertTokenizer.from_pretrained("bert-base-chinese", do_lower_case=True)


data_path="/hdfs/ipgsp/t-hasu/ppdata/zh-CN/"
output_path="/hdfs/ipgsp/t-hasu/ppdata/data-pretrain/"
if not os.path.exists(output_path):
    os.mkdir(output_path)
phones=set()
train=[]

def get_train(path, word):
    DOMTree = xml.dom.minidom.parse(path)
    collection = DOMTree.documentElement
    sis = collection.getElementsByTagName("si")
    for si in sis:
        text=""
        pho='_'

        # get the pronunciation
        def cut_sent(para):
            para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
            para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
            para = re.sub('(……)([^”’])', r"\1\n\2", para)  # 中文省略号
            para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
            # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
            para = para.rstrip()  # 段尾如果有多余的\n就去掉它
            # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
            return para.split("\n")
        for i,w in enumerate(si.getElementsByTagName("w")):
            text+=w.getAttribute('v')
        sents=cut_sent(text)
        train=train+sents

def get_train_ime(path, ime_len=1200000):
    with open(path,encoding='utf8') as f:
        for i,line in enumerate(f):
            if i%1000000==0:
                print(i)
            if i>=ime_len:
                break
            if i%4 == 0:
                text=line
            if i%4==1:
                pass
            if i%4==2:
                train.append(text)


#ime_words={dct[w] for w in ime_set}


# train
for word in sorted(os.listdir(data_path+"Annotation/")):
    print("Train set processing...", word)
    for file in os.listdir(data_path+"Annotation/"+word+"/trainingScript"):
        if file.split('.')[0]=="training":
            #print(file)
            get_train(data_path+"Annotation/"+word+"/trainingScript/"+file, word)

# IME
get_train_ime(data_path+"IMELogs/0-30000000.txt")

#save
with open(output_path+"/train.txt",'w',encoding='utf8') as f:
    f.write('\n'.join(train))
