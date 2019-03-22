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


for n,wpath in enumerate(wpaths):
    print("processing",wpath)
    #train data
    get_data(wlist[n], os.path.join(data_path,wpath,"training.xml"),'train')
    if os.path.exists(os.path.join(data_path,wpath,"training.News.xml")):
        get_data(wlist[n], os.path.join(data_path, wpath, "training.News.xml"), 'train')
    #test data
    get_data(wlist[n], os.path.join(data_path, wpath, "test.xml"), 'test')


#save
with open("../data/train.json",'w',encoding='utf8') as f:
    f.write(json.dumps(train))
with open("../data/test.json",'w',encoding='utf8') as f:
    f.write(json.dumps(test))
info={"words":wlist,"wpaths":plist,"phones":sorted(list(phones))}
with open("../data/info.json",'w',encoding='utf8') as f:
    f.write(json.dumps(info))
