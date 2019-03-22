import json
from xml.dom.minidom import parse
import xml.dom.minidom
import os


js={}
wlist=['背']
js['word']=wlist[0]
js['data']=[]
data_path="/bei/trainingScript/training.Xiaoice.xml"
tgt_path="bei.json"
#train data
DOMTree = xml.dom.minidom.parse(data_path)
collection = DOMTree.documentElement
sis = collection.getElementsByTagName("si")
phones=set()
for si in sis:
    js_data={'id':si.getAttribute('id')}
    js_data['text']=si.childNodes[1].childNodes[0].data
    js_data['position']=-1
    for i,w in enumerate(js_data['text']):
        if w==js['word']:
            js_data['position'] =i
            break
    assert js_data['position']!=-1
    for w in si.getElementsByTagName("w"):
        if w.getAttribute('v')==js['word']:
            js_data['phone']=w.getAttribute('p')
            break
    assert 'phone' in js_data.keys()
    js['data'].append(js_data)
    phones.add(js_data['phone'])
js['phones']=sorted(list(phones))

with open(tgt_path,'w') as f:
    f.write(json.dumps(js))
