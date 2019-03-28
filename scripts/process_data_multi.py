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
test_story=[]
test_news=[]
test_chat=[]
max_length_cut=80
words=set()
words_train=set()
test_list=[p[11:-4] for p in os.listdir(data_path+"TestCase/Story")]
train_list=os.listdir(data_path+"Annotation")

dct={}
def get_test(path,test):
    for word in os.listdir(path):
        print("Test set processing...", word)
        DOMTree = xml.dom.minidom.parse(path+word)
        collection = DOMTree.documentElement
        cases = collection.getElementsByTagName("case")
        dct[word] = cases[0].getAttribute('pron_polyword')
        for case in cases:
            js_data = {}
            js_data['text'] = case.getElementsByTagName("input")[0].childNodes[0].data.replace(' ', '')
            js_data['position'] = -1
            js_data['char'] = case.getAttribute('pron_polyword')
            for i, w in enumerate(js_data['text']):
                if w == case.getAttribute('pron_polyword'):
                    js_data['position'] = i
            # cut the text if too long
            if js_data['position'] > max_length_cut:
                # print(js_data['position'])
                js_data['text'] = js_data['text'][js_data['position'] - max_length_cut:]
                js_data['position'] = max_length_cut
            assert js_data['position'] != -1
            assert js_data['text'][js_data['position']] == case.getAttribute('pron_polyword')
            js_data['phone'] = [
                (js_data['char'], js_data['char'] + case.getElementsByTagName("part")[0].childNodes[0].data)]
            phones.add(js_data['phone'][0][1])
            words.add(js_data['char'])
            # assert ' ' not in js_data['text']
            test.append(js_data)

get_test(data_path+'TestCase/Story/',test_story)
get_test(data_path+'TestCase/News/',test_news)
get_test(data_path+'TestCase/ChitChat/',test_chat)

print(len(phones),sorted(list(phones)))
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
        phones_list=[]
        for w in si.getElementsByTagName("w"):
            if re.search('\s',w.getAttribute('v')):
                continue
            elif len(w.getAttribute('v')) != len(re.split('[-&]',w.getAttribute('p'))):
                print(w.getAttribute('v'),re.split('[-&]',w.getAttribute('p')))
                #min_len=min(len(w.getAttribute('v')),len(re.split('[-&]',w.getAttribute('p'))))
                js_data['text'] += w.getAttribute('v')
                #phones_list += re.split('[-&]',w.getAttribute('p'))[:min_len]
                #js_data['text'] += "_"
                phones_list += ["_"]*len(w.getAttribute('v'))
            else:
                js_data['text'] += w.getAttribute('v')
                phones_list += [p.strip() for p in re.split('[-&]',w.getAttribute('p'))]
        assert len(js_data['text'])==len(phones_list)
        if re.search('\s',js_data['text']):
            print(js_data['text'])
        for i in range(len(phones_list)):
            if js_data['text'][i] in words:
                words_train.add(js_data['text'][i])
                if js_data['text'][i]+phones_list[i] not in phones:
                    #print(js_data['text'][i],phones_list[i])
                    js_data['phone'].append((js_data['text'][i], '_'))
                else:
                    #phones.add(js_data['text'][i]+phones_list[i])
                    js_data['phone'].append((js_data['text'][i],js_data['text'][i]+phones_list[i]))

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
            print(file)
            get_train(data_path+"Annotation/"+word+"/trainingScript/"+file)

#phones.remove('_')
print(len(phones),sorted(list(phones)))
print(words-words_train)
print(len(train),len(test_story),len(test_news),len(test_chat))
#save
with open("../data/train.json",'w',encoding='utf8') as f:
    f.write(json.dumps(train))

with open("../data/test_story.json",'w',encoding='utf8') as f:
    f.write(json.dumps(test_story))
with open("../data/test_news.json",'w',encoding='utf8') as f:
    f.write(json.dumps(test_news))
with open("../data/test_chat.json",'w',encoding='utf8') as f:
    f.write(json.dumps(test_chat))

info={"words":test_list,"words_train":sorted(list(words_train)),"phones":sorted(list(phones))}
with open("../data/info.json",'w',encoding='utf8') as f:
    f.write(json.dumps(info))
