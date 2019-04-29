#!/usr/bin/env python
# coding=utf-8
import json
import re
from xml.dom.minidom import parse
import xml.dom.minidom
from pytorch_pretrained_bert.tokenization import BertTokenizer


data_path="/hdfs/ipgsp/t-hasu/ppdata/data-79-index/train.json"
add_path="/hdfs/ipgsp/t-hasu/ppdata/data-79-index/labeled_8352.xml"
output_path="/hdfs/ipgsp/t-hasu/ppdata/data-79-index/train_a.json"
with open(data_path,encoding='utf8') as f:
    train=json.loads(f.read())

from xml.dom.minidom import parse
import xml.dom.minidom
DOMTree = xml.dom.minidom.parse(add_path)
collection = DOMTree.documentElement
cases = collection.getElementsByTagName("case")
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese", do_lower_case=True)

for case in cases:
    index=int(case.getAttribute('index'))
    js={'char':'为'}
    js['text']=tokenizer.tokenize(case.getElementsByTagName("sent")[0].childNodes[0].data)
    if len(re.findall(js['char'],case.getElementsByTagName("sent")[0].childNodes[0].data))<index:
        index=len(re.findall(js['char'],case.getElementsByTagName("sent")[0].childNodes[0].data))
    count=1
    for i,c in enumerate(js['text']):
        if c==js['char'] and count==index:
            js['position']=i
            break
        elif c==js['char']:
            count+=1
    if js['position']>64:
        js['text']=js['text'][js['position']-64:]
        js['position']=64
    js['phone']=[[js['position'],js['char']+case.getElementsByTagName("pron")[0].childNodes[0].data]] 
    train.append(js)

with open(output_path,'w',encoding='utf8') as f:
    f.write(json.dumps(train,ensure_ascii=False))



