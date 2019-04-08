import json
with open("../data/train.json",encoding='utf8') as f:
    js=json.loads(f.read())
for i,j in enumerate(js):
    if len(j['text'])>128:
        print(i,j)
    if 'phone' not in j or not j['phone']:
        print(i,j)
