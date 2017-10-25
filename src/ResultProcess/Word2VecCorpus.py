from src.tools import Tools as tools
from src.tools import FileTools as ftools
import Dir
import os
from gensim.models import word2vec

def transfer(cleandata_root = Dir.res+"/cleandata_1189/news/",save_path = Dir.res+"/sen_data/1189_corpus.txt"):
    filelist = os.listdir(cleandata_root)
    lines = []
    for name in filelist:
        filepath = cleandata_root+name
        for line in ftools.read_lines(filepath):
            words = tools.seperate(line)
            for i in range(len(words)):
                if words[i].isdigit():
                    words[i] = "num"
            lines.append(' '.join(words))

    ftools.write_list(save_path,lines)

#
# def transfer_sen_corpus(cleandata_root = Dir.res+"/cleandata_604/news/",save_path = Dir.res+"/sen_data/604_sen_corpus.txt"):
#     filelist = os.listdir(cleandata_root)
#     lines = []
#     sen_index ={}
#     for name in filelist:
#         filepath = cleandata_root+name
#         for line in ftools.read_lines(filepath):
#             words = tools.seperate(line)
#             for i in range(len(words)):
#                 if words[i].isdigit():
#                     words[i] = "num"
#             if line not in sen_index.keys():
#                 sen_index[line] = "sen"+str(len(sen_index))
#             lines.append(' '.join(words)+" "+sen_index[line])
#     print(save_path+".txt")
#     ftools.write_dict(save_path+".txt",sen_index)
#     ftools.write_list(save_path,lines)


def train(path = Dir.res+"/sen_data/604_corpus.txt",save_path = Dir.res+"/w2v/w2v.model"):
    sentences = word2vec.Text8Corpus(path)  # 加载语料
    model = word2vec.Word2Vec(sentences, size=10,window=3,min_count=1)
    model.save(save_path)
    return save_path

def load_model(path = Dir.res+"/w2v/w2v.model"):
    model = word2vec.Word2Vec.load(path)
    return model

import math

def cos(x,y):
    xy = [x[i]*y[i] for i in range(len(x))]
    xx = [x[i]*x[i] for i in range(len(x))]
    yy = [y[i]*y[i] for i in range(len(y))]
    return sum(xy)/(math.sqrt(sum(xx))+math.sqrt(sum(yy)))

def cos_(x,y):
    import numpy as np
    import scipy.spatial.distance as distance
    a = np.array(x)
    b = np.array(y)
    c = 1 - distance.cosine(a, b)
    return c

# path = Dir.res+"/sen_data/604_sen_corpus.txt.txt"
# path1 = Dir.res+"/sen_data/604_sen_corpus.txt.txt"
# path2 = Dir.res+"/sen_data/604_sen_corpus.txt.txt"
# sen604 = ftools.read_dict(path)
# sen1073 = ftools.read_dict(path)
# sen1189 = ftools.read_dict(path)


# def get_sen_vec(model,sen,dict_path = Dir.res+"/sen_data/604_sen_corpus.txt.txt"):
#     sen604 = ftools.read_dict(dict_path,seperate="\t")
#     if sen in sen604.keys():
#         print(sen,sen604[sen])
#         return model.__getitem__(sen604[sen])
#     else:
#         print(sen)
#         return None

if __name__ == "__main__":
    # transfer_sen_corpus(cleandata_root= Dir.res+"/cleandata_1189/news/",save_path = Dir.res+"/sen_data/1189_sen_corpus.txt")
    train(path = Dir.res+"/sen_data/604_corpus.txt",save_path=Dir.res+"/w2v/w2v.model")
    model = load_model(path = Dir.res+"/w2v/w2v.model")
    # simi = model.most_similar("军分区",topn = 20)
    # print(model.__getitem__("困扰军事设施保护工作的主要有三大问题：核心要害军事设施安全环境恶化"))
    # print(model.__getitem__("联军"))
    #
    # print(cos(model.__getitem__("军分区"),model.__getitem__("联军")))
    # print(cos_(model.__getitem__("军分区"),model.__getitem__("联军")))

    # for var in simi:
    #     print(var[0],var[1])
    # pass

