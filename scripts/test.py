import json
with open("../data/train.json",encoding='utf8') as f:
    js=json.loads(f.read())
for i,j in enumerate(js):
    if i%100000==0:
        print(i)
    if len(j['phone'])<2:
        print(j)
    elif j['phone'][-1][0]>=128:
        print(i,j)
    elif 'phone' not in j or not j['phone'][-1][1]:
        print(i,j)
