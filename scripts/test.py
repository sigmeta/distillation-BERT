import json
from xml.dom.minidom import parse
import xml.dom.minidom
import os

data_path="../data/zh-CN/"
test_list=set(p[11:-4] for p in os.listdir(data_path+"TestCase/Story"))
train_list=set(os.listdir(data_path+"Annotation"))
words=test_list&train_list
words=list(words)
word=words[0]
DOMTree = xml.dom.minidom.parse(data_path+"TestCase/Story/ChildStory_"+word+".xml")
collection = DOMTree.documentElement
cases=collection.getElementsByTagName("case")
print(cases[0].getElementsByTagName("part")[0].childNodes[0].data)

DOMTree = xml.dom.minidom.parse(data_path+"Annotation/dang/trainingScript/training.News.xml")
collection = DOMTree.documentElement
sis = collection.getElementsByTagName("si")
print(sis[0].childNodes[1].childNodes[0].data)
