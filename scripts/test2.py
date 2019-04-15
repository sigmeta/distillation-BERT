#!/usr/bin/env python
# coding=utf-8
import json
path="/hdfs/ipgsp/t-hasu/ppdata/data-30M/test_chat.json"
with open(path,encoding='utf8') as f:
    js=json.loads(f.read())

for j in js:
    if j['char']=='哪':
        print(j)
