import json
from xml.dom.minidom import parse
import xml.dom.minidom
import os
import re
import sys
from pytorch_pretrained_bert.tokenization import BertTokenizer


tokenizer = BertTokenizer.from_pretrained("bert-base-chinese", do_lower_case=True)

stop={"'",'"',',','.','?','/','[',']','{','}','+','=','*','&','(',')','，','。','？',
      '“','”','’','‘','、','？','！','【','】','《','》','（','）','・','&quot;','——',
      '-','———',':','：','!','@','#','$','%','&',';','……','；','—','±'}
word2char={'ai2_ai1': '挨', 'ba3_ba4': '把', 'bao4_pu4': '曝', 'bei4_bei1': '背', 'ben1_ben4': '奔', 'bie2_bie4': '别',
           'bo1_bao1_pu1': '剥', 'bo2_bai3': '伯', 'bo2_bai3_bo4': '柏', 'bo2_bao2_bo4': '薄', 'bo2_po1': '泊',
           'bu2_bu4': '不', 'ceng2_zeng1': '曾', 'cha1_cha4_cha3_cha2': '叉', 'cha1_chai1_cha4_ci1_chai4': '差',
           'cha2_zha1': '查', 'chao2_zhao1': '朝', 'chen4_cheng1_cheng4': '称', 'cheng2_sheng4': '乘',
           'cheng4_cheng1_chen4': '秤', 'chong1_chong4': '冲', 'chong2_zhong4': '重', 'chu4_chu3': '处',
           'cuan2_zan3': '攒', 'da2_da1': '答', 'da3_da2': '打', 'dai4_dai1': '待', 'dan1_dan4': '担',
           'dang1_dang4': '当', 'dao3_dao4': '倒', 'de2_de4_dei3': '得', 'deng1_deng4': '蹬', 'di4_de4': '地',
           'di4_di1_di2': '的', 'diao4_tiao2_tiao4': '调', 'ding1_ding4': '钉', 'dou3_dou4': '斗', 'du1_dou1': '都',
           'du3_du4': '肚', 'du4_duo2': '度', 'e4_wu4_e3_wu1': '恶', 'fa4_fa1': '发', 'fang2_fang1': '坊',
           'fei1_fei3': '菲', 'fen1_fen4': '分', 'feng4_feng2': '缝', 'ga1_ga3_ga2': '嘎', 'gan3_gan1': '杆',
           'gan4_gan1': '干', 'gang3_gang1_gang4_gang2': '岗', 'ge1_ka3_luo1': '咯', 'ge2_ge1': '格',
           'geng1_geng4': '更', 'gong4_gong1': '供', 'gua1_gu1_pai4_gua3': '呱', 'guan1_guan4': '观',
           'guan4_guan1': '冠', 'hang2_xing2_heng2': '行', 'hao4_hao2': '号', 'hao4_hao3_hao1': '好',
           'he2_he4': '荷', 'he2_he4_huo4_huo2_hu2': '和', 'he2_hu2': '核', 'he4_he1': '喝',
           'heng4_heng2': '横', 'hong1_hong4_hong3': '哄', 'hou2_hou4': '侯', 'hu2_hu1_hu4': '糊',
           'hua2_hua4': '划', 'huan2_hai2': '还', 'huang4_huang3_huang1': '晃', 'hun2_hun4': '混',
           'ji3_ji1': '几', 'jia2_jia1_ga1': '夹', 'jia4_jia3': '假', 'jian1_jian4': '间', 'jiang4_jiang1_qiang1': '将',
           'jiao2_jue2_jiao4': '嚼', 'jiao3_jue2': '脚', 'jiao4_jiao1': '教', 'jiao4_jue2': '觉', 'jie2_jie1': '结',
           'jie3_jie4_xie4': '解', 'jin1_jin4': '禁', 'juan4_juan3': '卷', 'jue2_jiao3': '角', 'ka1_ga1': '咖',
           'ka1_ka3': '咔', 'ka3_qia3': '卡', 'kan4_kan1': '看', 'ke2_qiao4': '壳', 'kong1_kong4': '空',
           'la1_la3_la2_la4': '拉', 'le4_lei1': '勒', 'le4_yue4_lao4_yao4': '乐', 'le5_liao3': '了',
           'lei4_lei3_lei2': '累', 'liang2_liang4': '量', 'liang3_lia3': '俩', 'liang4_liang2': '凉',
           'liao2_liao1_liao4': '撩', 'lie3_lie1': '咧', 'lin2_lin1_lin4': '淋', 'liu4_liu1': '溜',
           'long2_long3': '笼', 'lou3_lou1_lou2': '搂', 'lu4_liu4': '陆', 'lu4_lou4': '露', 'lv4_shuai4': '率',
           'luo4_lao4_la4_luo1': '落', 'meng2_meng3_meng1': '蒙', 'mo2_mo4': '磨', 'mo2_mu2': '模', 'mo3_mo4_ma1': '抹',
           'mo4_mei2': '没', 'nan4_nan2': '难', 'nian2_zhan1': '粘', 'ning3_ning4_ning2': '拧', 'ning4_ning2': '宁',
           'pa2_ba1': '扒', 'pao4_pao1': '泡', 'pao4_pao2_bao1': '炮', 'pen4_pen1': '喷', 'pi1_pi3': '劈',
           'pian4_pian1': '片', 'piao1_piao4_piao3': '漂', 'pu4_bao4': '暴', 'pu4_pu1': '铺', 'qi2_ji1': '奇',
           'qiang2_jiang4_qiang3': '强', 'qiao4_qiao2': '翘', 'qie4_qie1': '切', 'qu3_qu1': '曲',
           'quan1_juan4_juan1': '圈', 'que4_qiao3_qiao1': '雀', 'sa3_sa1': '撒', 'sai4_se4_sai1': '塞',
           'san4_san3': '散', 'sang4_sang1': '丧', 'shan1_zha4_shi3': '栅', 'shan4_shan1': '扇', 'shao4_shao3': '少',
           'she4_she3': '舍', 'shen2_shi2': '什', 'sheng3_xing3': '省', 'sheng4_cheng2': '盛', 'shu2_shou2': '熟',
           'shu3_shu4_shuo4': '数', 'tan2_dan4': '弹', 'tang4_tang1': '趟', 'teng2_teng1': '腾', 'tiao1_tiao3': '挑',
           'tie3_tie4_tie1': '帖', 'tong1_tong4': '通', 'tu3_tu4': '吐', 'tuo4_ta4': '拓', 'wei4_wei2': '为',
           'xi4_ji4': '系', 'xia4_he4': '吓', 'xiang2_jiang4': '降', 'xiang4_xiang1': '相', 'xiao4_jiao4': '校',
           'xiao4_xiao1': '肖', 'xie3_xue4': '血', 'xing1_xing4': '兴', 'xiu3_xiu4_su4': '宿', 'xuan2_xuan4': '旋',
           'xue1_xiao1': '削', 'yan4_yan1': '燕', 'yan4_ye4_yan1': '咽', 'yi4_yi1_yi2': '一', 'yin1_yan1_yin3': '殷',
           'ying4_ying1': '应', 'yu3_yu4_yu2': '与', 'yue1_yao1': '约', 'yun4_yun1': '晕', 'za3_zha1_ze2': '咋',
           'zai3_zai4': '载', 'zai3_zi3_zi1': '仔', 'zang4_cang2': '藏', 'zang4_zang1': '脏', 'zao2_zuo4': '凿',
           'ze2_zhai2': '择', 'zha1_za1_zha2': '扎', 'zha2_zha4': '炸', 'zhan4_chan4': '颤', 'zhan4_zhan1': '占',
           'zhang3_chang2': '长', 'zhang3_zhang4': '涨', 'zhao3_zhua3': '爪', 'zhe2_she2_zhe1': '折', 'zhi1_zhi3': '只',
           'zhong3_zhong4_chong2': '种', 'zhong4_zhong1': '中', 'zhuan3_zhuan4_zhuai3': '转', 'zhuan4_chuan2': '传',
           'zhuo2_zhao2_zhao1': '着', 'zhuo2_zhu4_zhe1': '著', 'zu2_cu4': '卒', 'zuan1_zuan4': '钻',
           'zuo4_zuo1_zuo2': '作'}

#data_path="/blob/xuta/speech/tts/t-hasu/Polyphony/data/zh-CN/"
#output_path="/blob/xuta/speech/tts/t-hasu/ppdata/data-200-index/"
data_path=sys.argv[1]
output_path=sys.argv[2]
if not os.path.exists(output_path):
    os.mkdir(output_path)
phones=set()
train=[]
test_story=[]
test_news=[]
test_chat=[]
max_length_cut=64 # because the max length is 128 (with [CLS] and [SEP]), we should put the polyphonic character in the middle of the sentence.
max_length=126 # max length is 128 with [CLS] and [SEP], so we set 126 here.
words=set()
words_train=set()


trl=[]
tra=[]
p2=os.listdir(data_path)
for p in p2:
    fs=[]
    for f in os.listdir(os.path.join(data_path,p)):
        fs.append(f.split('.')[0])
    tra.append(p)
    if 'training' in fs:
        trl.append(p)
test_set=word2char.keys()
train_set=set(trl)
no_train=set([word2char[k] for k in word2char.keys()-train_set])
print("train set", train_set)
#assert not train_set-test_set

dup={k:set() for k in test_set}
trc={word2char[k]:0 for k in word2char}

def extract_train(path,char,the_list):
    # extract data from the file
    # the_list is the list we want to put the data in. [train, test_story, test_news, test_chat]
    DOMTree = xml.dom.minidom.parse(path)
    collection = DOMTree.documentElement
    sis = collection.getElementsByTagName("si")
    for si in sis:
        #text_now = si.getElementsByTagName("text")[0].childNodes[0].data.strip()
        js_data = {}
        js_data['text'] = []
        js_data['position'] = -1
        js_data['char'] = char
        js_data['phone']=[]
        pho = '_'
        # get the pronunciation
        for i, w in enumerate(si.getElementsByTagName("w")):
            if len(w.getAttribute('v'))==len(w.getAttribute('p').split('-')):
                for ii,c in enumerate(w.getAttribute('v')):
                    if c in no_train:
                        if len(js_data['text']) + ii<126 and trc[c]<10000:
                            pho = c + '\t' + w.getAttribute('p').split('-')[ii].strip()
                            js_data['phone'].append([len(js_data['text']) + ii, pho])
                            phones.add(pho)
                            trc[c]+=1 

            js_data['text'] += tokenizer.tokenize(w.getAttribute('v'))
        for p in js_data['phone']:
            if js_data['text'][p[0]]!=p[1][0]:
                print(p);js_data['phone'].remove(p);trc[p[1][0]]-=1
        if pho == '_':  # wrong case
            #print(js_data['text'])
            continue
        the_list.append(js_data)

def extract_data(path,char,the_list):
    # extract data from the file
    # the_list is the list we want to put the data in. [train, test_story, test_news, test_chat]
    DOMTree = xml.dom.minidom.parse(path)
    collection = DOMTree.documentElement
    sis = collection.getElementsByTagName("si")
    for si in sis:
        #text_now = si.getElementsByTagName("text")[0].childNodes[0].data.strip()
        js_data = {}
        js_data['text'] = ""
        js_data['position'] = -1
        js_data['char'] = char
        pho = '_'

        # get the pronunciation
        for i, w in enumerate(si.getElementsByTagName("w")):
            js_data['text'] += w.getAttribute('v')
            if w.getAttribute('v') == char:
                pho = js_data['char'] + '\t' + w.getAttribute('p')
            elif len(w.getAttribute('v'))>1 and char in w.getAttribute('v'):
                for ii,c in enumerate(w.getAttribute('v')):
                    if c==char:
                        pho = js_data['char'] + '\t' + w.getAttribute('p').split('-')[ii].strip()
        if pho == '_':  # wrong case
            print(js_data['text'])
            continue
        # get the position
        js_data['text'] = tokenizer.tokenize(js_data['text'])
        for i, w in enumerate(js_data['text']):
            if w == char:
                js_data['position'] = i
        # cut the text if too long
        if js_data['position'] > max_length_cut:
            js_data['text'] = js_data['text'][js_data['position'] - max_length_cut:]
            js_data['position'] = max_length_cut
        js_data['phone'] = [[js_data['position'], pho]]
        # assert js_data['position'] > -1
        # assert js_data['text'][js_data['position']] == char

        phones.add(js_data['phone'][-1][1])
        # js_data['text'] = ' '.join(js_data['text'])
        the_list.append(js_data)

def get_data(path, word):
    # get data from the path of a character
    if word in word2char:
        char = word2char[word]
    else:
        char='_'
    words_train.add(char)
    for f in os.listdir(path):
        #print("Processing...",f)
        if f.split('.')[0]=='training':
            if char!='_':
                extract_data(os.path.join(path,f),char,train)
            extract_train(os.path.join(path, f), char, train)
        elif word in test_set:
            if f.split('.')[1]=='Story':
                extract_data(os.path.join(path, f), char, test_story)
            elif f.split('.')[1]=='News':
                extract_data(os.path.join(path, f), char, test_news)
            else:
                extract_data(os.path.join(path, f), char, test_chat)




# test


# train
for word in sorted(tra):
    print("Processing...", word)
    get_data(os.path.join(data_path,word),word)
print(len(phones),sorted(list(phones)))
phones_train=phones.copy()

print(len(train),len(test_story),len(test_news),len(test_chat))

#save
with open(output_path+"/train.json",'w',encoding='utf8') as f:
    f.write(json.dumps(train, ensure_ascii=False))

with open(output_path+"/test_story.json",'w',encoding='utf8') as f:
    f.write(json.dumps(test_story, ensure_ascii=False))
with open(output_path+"/test_news.json",'w',encoding='utf8') as f:
    f.write(json.dumps(test_news, ensure_ascii=False))
with open(output_path+"/test_chat.json",'w',encoding='utf8') as f:
    f.write(json.dumps(test_chat, ensure_ascii=False))

print(trc)
info={"words_test":sorted(list(words)),
      "words_prepared":sorted(list(word2char[w] for w in word2char)),
      #"words_ime":sorted(list(ime_words)),
      "phones":sorted(list(phones)),
      "word2char":word2char,
      "train_count":trc}
with open(output_path+"/info.json",'w',encoding='utf8') as f:
    f.write(json.dumps(info, ensure_ascii=False))
