import json
from xml.dom.minidom import parse
import xml.dom.minidom
import os
import re


stop={"'",'"',',','.','?','/','[',']','{','}','+','=','*','&','(',')','，','。','？',
      '“','”','’','‘','、','？','！','【','】','《','》','（','）','・','&quot;','——',
      '-','———',':','：','!','@','#','$','%','&',';','……','；','—','±'}
data_path="../data/zh-CN/"
phones=set()
train=[]
test=[]
max_length_cut=80
words=set()
words_train=set()
test_list=[p[11:-4] for p in os.listdir(data_path+"TestCase/Story")]
train_list=os.listdir(data_path+"Annotation")

dct={}
for word in test_list:
    print("Test set processing...",word)
    DOMTree = xml.dom.minidom.parse(data_path+"TestCase/Story/ChildStory_"+word+".xml")
    collection = DOMTree.documentElement
    cases=collection.getElementsByTagName("case")
    dct[word]=cases[0].getAttribute('pron_polyword')
    for case in cases:
        js_data={}
        js_data['text']=case.getElementsByTagName("input")[0].childNodes[0].data.replace(' ','')
        js_data['position'] = -1
        js_data['char']=case.getAttribute('pron_polyword')
        for i, w in enumerate(js_data['text']):
            if w == case.getAttribute('pron_polyword'):
                js_data['position'] = i
        # cut the text if too long
        if js_data['position'] > max_length_cut:
            #print(js_data['position'])
            js_data['text'] = js_data['text'][js_data['position'] - max_length_cut:]
            js_data['position'] = max_length_cut
        assert js_data['position'] != -1
        assert js_data['text'][js_data['position']]==case.getAttribute('pron_polyword')
        js_data['phone']=['_']*len(js_data['text'])
        js_data['phone'][js_data['position']] = case.getElementsByTagName("part")[0].childNodes[0].data
        phones.add(js_data['phone'][0])
        words.add(js_data['char'])
        assert ' ' not in js_data['text']
        test.append(js_data)


def get_train(path):
    DOMTree = xml.dom.minidom.parse(path)
    char = '_'
    collection = DOMTree.documentElement
    sis = collection.getElementsByTagName("si")
    for si in sis:
        js_data = {}
        js_data['text'] = ""
        js_data['position'] = -1
        js_data['char'] = char
        js_data['phone']=[]
        for w in si.getElementsByTagName("w"):
            if w.getAttribute('v')==' ':
                continue
            elif w.getAttribute('v') in stop:
                js_data['text']+=w.getAttribute('v')[0]
                js_data['phone'] += ['_']
            elif len(w.getAttribute('v')) != len(re.split('[-&]',w.getAttribute('p'))):
                js_data['text'] += w.getAttribute('v')[0]
                js_data['phone'] += ['_']
            elif w.getAttribute('v') in words:
                words_train.add(w.getAttribute('v'))
                js_data['text'] += w.getAttribute('v')
                js_data['phone'] += [w.getAttribute('p')]
            else:
                js_data['text'] += w.getAttribute('v')
                #js_data['phone'] += [p.strip() for p in re.split('[-&]',w.getAttribute('p'))]
                js_data['phone'] += ['_']*len(w.getAttribute('v'))
        if len(js_data['text'])!=len(js_data['phone']):
            print(js_data['text'])
            print(js_data['phone'])
        assert ' ' not in js_data['text']
        for p in js_data['phone']:
            phones.add(p)

        train.append(js_data)
        # cut long sentences
        '''
        else:
            texts_list = js_data['text'].split('。')
            if len(texts_list[-1]) < 2:
                texts_list = texts_list[:-1]
            phones_list=[]
            n=0
            for t in texts_list:
                phones_list.append(js_data['phone'][n:n+len(t)])
                n+=len(t)
            print(texts_list)
            print(phones_list)
            for i in range(len(texts_list)):
                js_data['text']=texts_list[i]
                js_data['phone']=phones_list[i]
                train.append(js_data)
        '''

for word in train_list:
    print("Train set processing...", word)
    for file in os.listdir(data_path+"Annotation/"+word+"/trainingScript"):
        if file.split('.')[0]=="training":
            get_train(data_path+"Annotation/"+word+"/trainingScript/"+file)

phones.remove('_')
print(len(phones),sorted(list(phones)))
print(words-words_train)
print(len(train),len(test))
#save
with open("../data/train.json",'w',encoding='utf8') as f:
    f.write(json.dumps(train))
with open("../data/test.json",'w',encoding='utf8') as f:
    f.write(json.dumps(test))
info={"words":test_list,"words_train":sorted(list(words_train)),"phones":sorted(list(phones))}
with open("../data/info.json",'w',encoding='utf8') as f:
    f.write(json.dumps(info))
