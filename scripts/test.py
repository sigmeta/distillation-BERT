import json
with open("data/train.json",encoding='utf8') as f:
    js=json.loads(f.read())
for i,j in enumerate(js):
    if i%10000000==0:
        print(i)
    for pos,pro in j['phone']:
        if pro=='调ti aa_h o_l':
            print(j)
