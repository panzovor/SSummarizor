import os
import jieba
import re
from src.tools import FileTools as ftools
import jieba.posseg as pog
from urllib import request as request
from urllib import parse as parse
import time
import Dir
import pickle
import hashlib
from enum import Enum
import math

stop_words =set()
sw = ftools.read_lines(Dir.res+"/parameter/segmentation/stopwords.txt")
for var in sw:
    stop_words.add(var)

def normarlize(value_list):
    min_value = min(value_list)
    max_value = max(value_list)
    for i in range(len(value_list)):
        value_list[i] = (value_list[i]-min_value)/(max_value-min_value)

def sim(sen1,sen2):
    w1 = seperate(sen1)
    w2 = seperate(sen2)
    count =0
    for w in w1:
        if w in w2:
            count+=1
    return count

def md5(text):
    pair    = 10
    gap = int(len(text)/pair)
    tmp = [''.join(text)[var*gap] for var in range(pair)]
    tmp = ''.join(tmp)
    tmp = tmp.replace("/","-")
    tmp = tmp.replace("\\","-")
    return tmp

def vector_add(v1,v2):
    tmp = []
    for  i in range(len(v1)):
        tmp.append(v1[i]+v2[i])
    return tmp

def vector_add_multi(vec_list):
    if len(vec_list) == 0:
        return vec_list
    if len(vec_list)==1:
        return vec_list[0]
    tmp = vector_add(vec_list[0],vec_list[1])
    for i in range(2,len(vec_list)):
        tmp = vector_add(tmp,vec_list[i])
    return tmp

def vector_multi(v1,num_v):
    for i in range(len(v1)):
        v1[i] *= num_v
    return v1

class ltp():
    def __init__(self):
        self.last_time = time.time()
        self.count = 0

    def test_ltp(self,sentences):
        mykey = "L1L3F8L7i0WD1GtgiaQKvHNQxxgQqdJWsTKBwXwC"
        url_get_base = "http://api.ltp-cloud.com/analysis/"
        args = {
            'api_key' : mykey,
            'text' : sentences,
            'pattern' : 'sdp',
            'format' : 'plain'
        }
        now_time = time.time()
        if now_time - self.last_time < 1/200:
            # print("wait",now_time-self.last_time,self.last_time,now_time)
            time.sleep(1/200-now_time+self.last_time)
        self.last_time = now_time
        self.count+=1
        # urllib.parse.urlencode(values).encode(encoding='UTF8')

        result = request.urlopen(url_get_base, parse.urlencode(args).encode(encoding="utf-8")) # POST method
        content = result.read().strip().decode(encoding = "utf-8")
        return content

    def short_sentences(self,sentences):
        # target_mark = ["VOB","COO","SBV","IOB","FOB","HED","IS","WP"]
        target_mark = ["Nmod", "Mann", "Feat", "Tmod", "ePrec", "eProg", "mTime"]
        # while 1:
        #     try:
        content = self.test_ltp(sentences)
                # break
            # except:
            #     print("error when ",self.count)
            #     time.sleep(0.5)
        tmp = {}
        for line in content.split("\n"):
            line_seperate = line.split(" ")
            if line_seperate[-1] not in target_mark:
                for var in line_seperate[:2]:
                    if "_" in var:
                        var_s = var.split("_")
                        tmp[var_s[0]] = int(var_s[1])

        tmp_item = sorted(tmp.items(), key=lambda d: d[1])
        short_sen = []
        for var, index in tmp_item:
            short_sen.append(var)
        return short_sen


def ltp_syntactic_dependency(sentences):
    mykey = "L1L3F8L7i0WD1GtgiaQKvHNQxxgQqdJWsTKBwXwC"
    url_get_base = "http://api.ltp-cloud.com/analysis/"
    args = {
        'api_key' : mykey,
        'text' : sentences,
        'pattern' : 'sdp',
        'format' : 'plain'
    }

    # urllib.parse.urlencode(values).encode(encoding='UTF8')
    result = request.urlopen(url_get_base, parse.urlencode(args).encode(encoding="utf-8")) # POST method
    content = result.read().strip().decode(encoding = "utf-8")
    return content


def seperate(sentence):
    chinese = judge(sentence)
    # print(sentence,chinese)
    if not chinese:
        if " " in sentence:
            return sentence.split(" ")
        else:
            return [sentence]
    else:
        tmp = list(jieba.cut(sentence))
        # tmp = seperate(sentence)
        words = []
        for t in tmp:
            t = t.strip()
            if t == "" or  t in stop_words:
                continue
            words.append(t)
        return words

def sen_pog(sen):
    tmp = pog.cut(sen)
    return tmp

def seperate_pog(sen):
    words,tags =[],[]
    for tmp in sen_pog(sen):
        words.append(tmp.word)
        tags.append(tmp.flag)
    return words,tags

def save_object(obj, path):
    pickle.dump(obj, open(path, mode="wb"))


def load_object(path):
    return pickle.load(open(path, mode="rb"))

def seperate_sentences(essay):
    sentences = []
    if not isinstance(essay, list):
        regex = "！|？|。|；|\.\.\.\.\.\."
        essay = essay.replace("\n","")
        new_sentences = re.split(regex, essay)
        # print(sentences.__len__())
        for sen in new_sentences:
            if sen.strip().__len__() > 3:
                sentences.append(sen.strip())
    else:
        # print("wrong  ")
        for sen in essay:
            if sen.strip().__len__()>3:
                sentences.append(sen.strip())
    return sentences

def judge(string):

    zhPattern = re.compile(u'[\u4e00-\u9fa5]+')
    match = zhPattern.search(string)
    if match != None:
        return True
    else:
        return False

def get_files (fileroot):
    return os.listdir(fileroot)

def short_sentences(sentences):
    # target_mark = ["VOB","COO","SBV","IOB","FOB","HED","IS","WP"]
    target_mark = ["Nmod","Mann","Feat","Tmod","ePrec","eProg","mTime"]
    content = ltp_syntactic_dependency(sentences)
    tmp = {}
    for line in content.split("\n"):
        line_seperate = line.split(" ")
        if line_seperate[-1] not in target_mark:
            for var in line_seperate[:2]:
                if "_" in var:
                    var_s = var.split("_")
                    tmp[var_s[0]] = int(var_s[1])

    tmp_item = sorted(tmp.items(), key = lambda d:d[1] )
    short_sen = []
    for var,index in tmp_item:
        short_sen.append(var)
    return short_sen


class Distance(Enum):

    COS =0
    EUD =1
    OCC =2
    OCCLOSE =3


class Dist():

    def __init__(self):
        self.dis = Distance.COS

    def sim(self, sim1, sim2,dis = None):
        if len(sim1) != len(sim2):
            return 0

        if dis!=None:
            self.dis = dis

        # print("(dis",self.dis)

        if self.dis == Distance.COS:
            xy = [sim1[i] * sim2[i] for i in range(len(sim1))]
            xx = [sim1[i] * sim1[i] for i in range(len(sim1))]
            yy = [sim2[i] * sim2[i] for i in range(len(sim1))]
            tmp = sum(xx) * sum(yy)
            if tmp == 0:
                return 0
            return sum(xy) / math.sqrt(tmp)

        if self.dis == Distance.EUD:
            dxy = [(sim1[i] - sim2[i]) ** 2 for i in range(len(sim1))]
            dxy = math.sqrt(sum(dxy))
            return 1 / (1 + dxy)

        if self.dis == Distance.OCC:
            count,countl,countr =0,0,0
            for i in range(len(sim1)):
                if sim1[i] == sim2[i]:
                    count+=1
                countl+=sim1[i]
                countr+=sim2[i]
            return count/max(countl,countr)

        if self.dis == Distance.OCCLOSE:
            count = 0
            for i in range(len(sim1)):
                if sim1[i]>0 and sim2[i]>0:
                    count+=1
            return count / len(sim1)

if __name__ == "__main__":
    # vec1 = [1,1,1,1,1]
    # vec2 = [1,1,1,1,1]
    #
    # vects = [[1,1,1,1,1],
    #          [1,1,1,1,1],
    #          [1,1,1,1,1],
    #          [1,1,1,1,1],
    #          [1,1,1,1,1]]
    # print("tsete")
    # print(vector_add(vec1,vec2))
    # print(vector_multi(vector_add_multi(vects),1/len(vects)))
    # print(vector_add_multi(vects))
    sens ="近日，教育部官方网站发布“关于宣布失效一批规范性文件的通知"
    print(seperate_pog(sens))