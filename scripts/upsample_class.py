import json
import math

# balance only within every character
data_path="/hdfs/ipgsp/t-hasu/ppdata/data-79/"
freq={}
cate={}
with open(data_path+"train.json",encoding='utf8') as f:
    js=json.loads(f.read())

for data in js:
    phone=data['phone']
    if phone[0] in freq:
        if phone in freq[phone[0]]:
            freq[phone[0]][phone]+=1
        else:
            freq[phone[0]][phone]=1
    else:
        freq[phone[0]]={phone:1}
    if phone in cate:
        cate[phone].append(data)
    else:
        cate[phone]=[data]

res=[]
for char in freq:
    maxnum=max([freq[char][phone] for phone in freq[char]])
    for phone in freq[char]:
        if freq[char][phone]<maxnum:
            mul=int(math.sqrt(maxnum/freq[char][phone]))
            res+=cate[phone]*mul
        else:
            res+=cate[phone]

with open(data_path+"train_up_class.json",'w',encoding='utf8') as f:
    f.write(res)
