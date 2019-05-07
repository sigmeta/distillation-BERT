import json
from xml.dom.minidom import parse
import xml.dom.minidom
import os
import re
from pytorch_pretrained_bert.tokenization import BertTokenizer


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

data_path="/hdfs/ipgsp/t-hasu/ppdata/en-US/"
output_path="/hdfs/ipgsp/t-hasu/ppdata/en-79/"
if not os.path.exists(output_path):
    os.mkdir(output_path)
phones=set()
train=[]
test=[]
max_length_cut=64
max_length=126
words=set()
words_train=set()
train_set=set()

dct={}


def get_test(path):
    for word in os.listdir(path):
        if not os.path.exists(os.path.join(path,word,'testing.xml')):
            print("No test file for ",word)
            continue
        print("Test set processing...", word)
        DOMTree = xml.dom.minidom.parse(os.path.join(path,word,'testing.xml'))
        char = word
        words_train.add(char)
        collection = DOMTree.documentElement
        sis = collection.getElementsByTagName("si")
        for si in sis:
            js_data = {}
            js_data['text'] = ""
            js_data['position'] = -1
            js_data['char'] = char
            pho = '_'

            # get the pronunciation
            for i, w in enumerate(si.getElementsByTagName("w")):
                js_data['text'] = js_data['text'] + w.getAttribute('v') + ' '
                if w.getAttribute('v') == char:
                    pho = js_data['char'] + '\t' + w.getAttribute('p')
            if pho == '_':  # wrong case
                print(js_data['text'])
                continue
            # get the position
            js_data['text'] = tokenizer.tokenize(js_data['text'])
            for i, w in enumerate(js_data['text']):
                if w == char:
                    js_data['position'] = i
            # cut the text if too long
            if js_data['position'] > max_length_cut and len(js_data['text'])>max_length:
                js_data['text'] = js_data['text'][js_data['position'] - max_length_cut:]
                js_data['position'] = max_length_cut
            js_data['phone'] = [[js_data['position'], pho]]
            # assert js_data['position'] > -1
            # assert js_data['text'][js_data['position']] == char

            phones.add(js_data['phone'][-1][1])
            # js_data['text'] = ' '.join(js_data['text'])
            test.append(js_data)


def get_train(path):
    for word in os.listdir(path):
        if not os.path.exists(os.path.join(path,word,'trainingScript/training.xml')):
            print("No training file for ",word)
            continue
        else:
            train_set.add(word)
        print("Training set processing...", word)
        DOMTree = xml.dom.minidom.parse(os.path.join(path,word,'trainingScript/training.xml'))
        char = word
        words_train.add(char)
        collection = DOMTree.documentElement
        sis = collection.getElementsByTagName("si")
        for si in sis:
            js_data = {}
            js_data['text'] = ""
            js_data['position'] = -1
            js_data['char'] = char
            pho='_'

            # get the pronunciation
            for i,w in enumerate(si.getElementsByTagName("w")):
                js_data['text']=js_data['text']+w.getAttribute('v')+' '
                if w.getAttribute('v') == char:
                    pho=js_data['char'] + '\t' + w.getAttribute('p')
            if pho=='_': # wrong case
                print(js_data['text'])
                continue
            # get the position
            js_data['text'] = tokenizer.tokenize(js_data['text'])
            for i,w in enumerate(js_data['text']):
                if w==char:
                    js_data['position']=i
            # cut the text if too long
            if js_data['position'] > max_length_cut and len(js_data['text'])>max_length:
                js_data['text'] = js_data['text'][js_data['position'] - max_length_cut:]
                js_data['position'] = max_length_cut
            js_data['phone'] = [[js_data['position'], pho]]
            #assert js_data['position'] > -1
            #assert js_data['text'][js_data['position']] == char

            phones.add(js_data['phone'][-1][1])
            #js_data['text'] = ' '.join(js_data['text'])
            train.append(js_data)


# test
get_test(data_path)
print(dct)

print(len(phones),sorted(list(phones)))
phones_test=phones.copy()

# train
get_train(data_path)
print(len(phones),sorted(list(phones)))
phones_train=phones.copy()

print(sorted(list(phones_train-phones_test)))
#print(sorted(list(phones_ime-phones_train-phones_test)))

print(words-words_train)
print(len(train),len(test))

#save
with open(output_path+"/train.json",'w',encoding='utf8') as f:
    f.write(json.dumps(train, ensure_ascii=False))

with open(output_path+"/test.json",'w',encoding='utf8') as f:
    f.write(json.dumps(test, ensure_ascii=False))

info={"words_test":sorted(list(words)),
      "words_prepared":sorted(list(dct[w] for w in train_set)),
      #"words_ime":sorted(list(ime_words)),
      "phones":sorted(list(phones))}
with open(output_path+"/info.json",'w',encoding='utf8') as f:
    f.write(json.dumps(info, ensure_ascii=False))
