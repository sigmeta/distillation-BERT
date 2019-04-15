import json
import math

# balance only within every character
data_path="/hdfs/ipgsp/t-hasu/ppdata/data-79/"
freq={}
cate={}
with open(data_path+"train.json",encoding='utf8') as f:
    js=json.loads(f.read())

for data in js:
    char=data['phone'][0]
    if char in freq:
        freq[char]+=1
    else:
        freq[char]=1
    if char in cate:
        cate[char].append(data)
    else:
        cate[char]=[data]

res=[]
maxnum=max([freq[char] for char in freq])
for char in freq:
    if freq[char]<maxnum:
        mul=int(maxnum/freq[char])
        res+=cate[char]*mul
    else:
        res+=cate[char]

with open(data_path+"train_up_char.json",'w',encoding='utf8') as f:
    f.write(res)
