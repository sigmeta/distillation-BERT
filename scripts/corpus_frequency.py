import json
from tqdm import tqdm
from pytorch_pretrained_bert.tokenization import BertTokenizer


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
freq={k:0 for k in tokenizer.vocab}

corpus_file='/var/storage/shared/ipgsp/t-hasu/polyphone/a/en_corpus/en_corpus.txt'
output_file='/var/storage/shared/ipgsp/t-hasu/polyphone/a/en_corpus/frequency.json'
with open(corpus_file) as f:
    for line in tqdm(f):
        tokens=tokenizer.tokenize(line.strip())
        for t in tokens:
            freq[t]+=1

with open(output_file,'w') as f:
    f.write(json.dumps(freq,indent=2))
