import json
from xml.dom.minidom import parse
import xml.dom.minidom
import os


wlist=['背','差','长','传','当','地','得','行','觉','空','乐','难','片','弹','为','系','相','血','应','只','重','中']
plist=['bei','cha','chang','chuan','dang','de','dei','hang','jiao','kong','le','nan','pian','tan','wei','xi','xiang','xue','ying','zhi','zhong','zhong1']
data_path="../data/zh-CN/"
wpaths=plist
phones=set()
train=[]
test=[]
max_length_cut=80
def get_data(char, path, split):
    DOMTree = xml.dom.minidom.parse(path)
    collection = DOMTree.documentElement
    sis = collection.getElementsByTagName("si")
    for si in sis:
        js_data = {'id': si.getAttribute('id')}
        js_data['text'] = si.childNodes[1].childNodes[0].data
        js_data['position'] = -1
        js_data['char']=char
        for i, w in enumerate(js_data['text']):
            if w == char:
                js_data['position'] = i
        # cut the text if too long
        if js_data['position'] > 500:
            js_data['text'] = js_data['text'][js_data['position'] - 400:]
            js_data['position'] = 400
        assert js_data['position'] != -1
        assert js_data['text'][js_data['position']]==char
        for w in si.getElementsByTagName("w"):
            if w.getAttribute('v') == char:
                js_data['phone'] = w.getAttribute('p')
        assert 'phone' in js_data.keys()
        phones.add(js_data['phone'])
        if split=="train":
            train.append(js_data)
        elif split=="test":
            test.append(js_data)
        else:
            raise Exception("split should be in ['train','test']")


test_list=set(p[11:-4] for p in os.listdir(data_path+"TestCase/Story"))
train_list=set(os.listdir(data_path+"Annotation"))
words=test_list&train_list
dct={}
for word in words:
    print("Test set processing...",word)
    DOMTree = xml.dom.minidom.parse(data_path+"TestCase/Story/ChildStory_"+word+".xml")
    collection = DOMTree.documentElement
    cases=collection.getElementsByTagName("case")
    dct[word]=cases[0].getAttribute('pron_polyword')
    for case in cases:
        js_data={}
        js_data['text']=case.getElementsByTagName("input")[0].childNodes[0].data
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
        js_data['phone'] = js_data['char']+case.getElementsByTagName("part")[0].childNodes[0].data
        phones.add(js_data['phone'])
        test.append(js_data)

for word in words:
    print("Train set processing...", word)
    DOMTree = xml.dom.minidom.parse(data_path+"Annotation/"+word+"/trainingScript/training.Story.xml")
    char=dct[word]
    collection = DOMTree.documentElement
    sis = collection.getElementsByTagName("si")
    for si in sis:
        js_data = {'id': si.getAttribute('id')}
        js_data['text'] = si.childNodes[1].childNodes[0].data
        js_data['position'] = -1
        js_data['char']=char
        for i, w in enumerate(js_data['text']):
            if w == char:
                js_data['position'] = i
        # cut the text if too long
        if js_data['position'] > max_length_cut:
            #print(js_data['position'])
            js_data['text'] = js_data['text'][js_data['position'] - max_length_cut:]
            js_data['position'] = max_length_cut
        assert js_data['position'] != -1
        assert js_data['text'][js_data['position']]==char
        for w in si.getElementsByTagName("w"):
            if w.getAttribute('v') == char:
                js_data['phone'] = js_data['char']+w.getAttribute('p')
        assert 'phone' in js_data.keys()
        #assert js_data['phone'] in phones
        if js_data['phone'] not in phones:
            print(js_data['phone'])
            phones.add(js_data['phone'])
        train.append(js_data)

    if os.path.exists(data_path + "Annotation/" + word + "/trainingScript/training.Story_R2.xml"):
        DOMTree = xml.dom.minidom.parse(data_path + "Annotation/" + word + "/trainingScript/training.Story_R2.xml")
        char = dct[word]
        collection = DOMTree.documentElement
        sis = collection.getElementsByTagName("si")
        for si in sis:
            js_data = {'id': si.getAttribute('id')}
            js_data['text'] = si.childNodes[1].childNodes[0].data
            js_data['position'] = -1
            js_data['char'] = char
            for i, w in enumerate(js_data['text']):
                if w == char:
                    js_data['position'] = i
            # cut the text if too long
            if js_data['position'] > max_length_cut:
                # print(js_data['position'])
                js_data['text'] = js_data['text'][js_data['position'] - max_length_cut:]
                js_data['position'] = max_length_cut
            assert js_data['position'] != -1
            assert js_data['text'][js_data['position']] == char
            for w in si.getElementsByTagName("w"):
                if w.getAttribute('v') == char:
                    js_data['phone'] = js_data['char'] + w.getAttribute('p')
            assert 'phone' in js_data.keys()
            # assert js_data['phone'] in phones
            if js_data['phone'] not in phones:
                print(js_data['phone'])
                phones.add(js_data['phone'])
            train.append(js_data)

print(len(phones),phones)

#save
with open("../data/train.json",'w',encoding='utf8') as f:
    f.write(json.dumps(train))
with open("../data/test.json",'w',encoding='utf8') as f:
    f.write(json.dumps(test))
info={"words":wlist,"wpaths":plist,"phones":sorted(list(phones))}
with open("../data/info.json",'w',encoding='utf8') as f:
    f.write(json.dumps(info))
