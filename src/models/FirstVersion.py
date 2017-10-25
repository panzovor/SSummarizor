import time
import math
import random
from enum import Enum

import networkx as nx
import numpy as np
from sklearn.cluster import KMeans

# from src.models import AutoCoder as Auto
import Dir
from src.ResultProcess import Word2VecCorpus as wvc
from src.tools import FileTools as ftools
from src.tools import Tools as tools
from src.vectorizer.words_bag_vector import words_bag_vector as MyVec

class Distance(Enum):

    COS =0
    EUD =1
    JAC =2

class Simple_SenVec():

    def __init__(self):
        self.name = "Simple_SenVec"

    def vectorize(self,text):
        sentences = tools.seperate_sentences(text)
        res =[]
        words = {}
        sen_w = []
        for i in range(len(sentences)):
            sen_words = tools.seperate(sentences[i])
            sen_w.append(sen_words)
            for w in sen_words:
                if w  not in words.keys():
                    words[w] = len(words)
        for i in range(len(sen_w)):
            tmp =[0]*len(words)
            for var in sen_w[i]:
                tmp[words[var]]+=1
            res.append(tmp)
        return res

class Word2Vec():

    def __init__(self,weighted = True):
        self.model = wvc.load_model()

        self.vec_length = 10
        self.tr = TextRank()
        self.weighted=  weighted
        if self.weighted:
            self.name = "Word2Vec weighted"
        else:
            self.name = "Word2Vec unweighted"

    def vectorize(self,text):
        if self.weighted:
            return self.weighted_vectorize(text)
        else:
            return self.unweighted_vectorize(text)

    def unweighted_vectorize(self,text):
        res =[]
        sentences = tools.seperate_sentences(text)
        for line in sentences:
            tmp = []
            for word in tools.seperate(line):
                if self.model.wv.vocab.__contains__(word):
                    wv = self.model.__getitem__(word)
                    tmp.append(wv)
                else:
                    tmp.append([0]* self.vec_length)
            tmp = tools.vector_add_multi(tmp)
            tmp = tools.vector_multi(tmp,1/(len(tmp)))
            res.append(tmp)
        return res

    def weighted_vectorize(self,text):
        res =[]
        sentences = tools.seperate_sentences(text)
        tr_text = self.tr.textrank(text)
        for sen in sentences:
            tmp =[]
            tmp_weight =[]
            sen_words = tools.seperate(sen)
            for w in sen_words:
                if self.model.wv.vocab.__contains__(w):
                    tmp.append(self.model.__getitem__(w))
                    if w in tr_text:
                        tmp_weight.append(tr_text[w])
                    else:
                        tmp_weight.append(1/len(sen_words))
                else:
                    tmp.append([0]* self.vec_length)
                    tmp_weight.append(1/len(sen_words))
            for i in range(len(tmp)):
                tmp[i] = tools.vector_multi(tmp[i],tmp_weight[i]/sum(tmp_weight))

            sen_vec = tools.vector_add_multi(tmp)
            if len(sen_vec) == 0:
                print(sen)
            res.append(sen_vec)
        return res

class VecSyntacticDependency():
    def __init__(self):
        self.tr = TextRank()
        self.name  ="VecSyntacticDependency "
        self.ltp = tools.ltp()

    def vectorize(self,text):
        sens = tools.seperate_sentences(text)
        short_text =[]
        for sen in sens:
            short_text.append( self.ltp.short_sentences(sen))
        s_w_tr = self.tr.textrank_matrix(short_text)
        sen_vs=  []
        for sen in sens:
            tmp =[]
            for w in s_w_tr.keys():
                if w in sen:
                    tmp.append(s_w_tr[w])
                else:
                    tmp.append(0.0)
            sen_vs.append(tmp)
        return sen_vs

class NVVector():
    def __init__(self):
        self.name = "noun and verb vector"
        self.tr = TextRank()

    def vectorize(self,text):
        sens = tools.seperate_sentences(text)
        matrix = []
        for sen in sens:
            tmp = tools.sen_pog(sen)
            pog_tmp = []
            for w,p in tmp:
                if p == "n" or "v" in p:
                    pog_tmp.append(w)
            matrix.append(pog_tmp)
        tr_res =self.tr.textrank_matrix(matrix)
        # for w in tr_res

class TextRank():

    def textrank(self, text):
        sentences = tools.seperate_sentences(text)
        words = {}
        words_list = []
        res = {}
        sen_words = []
        for sen in sentences:
            ws = tools.seperate(sen)
            sen_words.append(ws)
            for w in ws:
                if w not in words.keys():
                    words_list.append(w)
                    words[w] = len(words)
        matrix = np.zeros((len(words),len(words)))
        # matrix = [[0] * len(words) for var in range(len(words))]
        for sen_w in sen_words:
            for i in range(len(sen_w)):
                for j in range(i, len(sen_w)):
                    # print(words[sen_w[i]],words[sen_w[j]],len(words))
                    matrix[words[sen_w[i]],words[sen_w[j]]] += 1
                    matrix[words[sen_w[j]],words[sen_w[i]]] += 1
        nx_graph = nx.from_numpy_matrix(matrix)
        nx_parameter = {'alpha': 0.85}
        score = nx.pagerank(nx_graph, **nx_parameter)
        sorted_score = sorted(score.items(), key=lambda item: item[1], reverse=True)
        for index, value in sorted_score:
            if words_list[index] not in res.keys():
                res[words_list[index]] = value
        return res


    def get_similarity_(self,sen1,sen2):
        word_list1 = tools.seperate(sen1)
        word_list2 = tools.seperate(sen2)

        words = list(set(word_list1 + word_list2))
        vector1 = [float(word_list1.count(word)) for word in words]
        vector2 = [float(word_list2.count(word)) for word in words]

        vector1, vector2 = np.mat(vector1), np.mat(vector2)
        dist = np.sqrt(np.sum(np.square(vector1 - vector2)))
        return dist


    def textrank_sen(self,sentences):
        graph_array = np.zeros((sentences.__len__(), sentences.__len__()))

        nx_parameter = {'alpha': 0.85}

        for x in range(graph_array.__len__()):
            for y in range(x, graph_array[x].__len__()):
                graph_array[x, y] = self.get_similarity_(sentences[x], sentences[y])
                graph_array[y, x] = graph_array[x, y]
        nx_graph = nx.from_numpy_matrix(graph_array)
        score = nx.pagerank(nx_graph, **nx_parameter)
        return score

    def textrank_matrix(self,sen_words):
        words = {}
        words_list = []
        res = {}
        for sen in sen_words:
            for w in sen:
                if w not in words.keys():
                    words_list.append(w)
                    words[w] = len(words)
        matrix = np.zeros((len(words), len(words)))
        # matrix = [[0] * len(words) for var in range(len(words))]
        for sen_w in sen_words:
            for i in range(len(sen_w)):
                for j in range(i, len(sen_w)):
                    # print(words[sen_w[i]],words[sen_w[j]],len(words))
                    matrix[words[sen_w[i]], words[sen_w[j]]] += 1
                    matrix[words[sen_w[j]], words[sen_w[i]]] += 1
        nx_graph = nx.from_numpy_matrix(matrix)
        nx_parameter = {'alpha': 0.85}
        score = nx.pagerank(nx_graph, **nx_parameter)
        sorted_score = sorted(score.items(), key=lambda item: item[1], reverse=True)
        for index, value in sorted_score:
            if words_list[index] not in res.keys():
                res[words_list[index]] = value
        return res

class Dist():

    def __init__(self):
        self.dis = Distance.COS

    def sim(self, sim1, sim2,dis = None):
        if len(sim1) != len(sim2):
            return 0

        if dis!=None:
            self.dis = dis

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

        if self.dis == Distance.JAC:
            xy = [sim1[i] * sim2[i] for i in range(len(sim1))]
            xx = [sim1[i] * sim1[i] for i in range(len(sim1))]
            yy = [sim2[i] * sim2[i] for i in range(len(sim1))]
            tmp = math.sqrt(sum(xx)) + math.sqrt(sum(yy)) - xy
            if tmp == 0:
                return 0
            return xy / tmp

class Cluster():


    def __init__(self):
        pass

    def kmeans(self,data,k=3,random_start = True):
        npdata = np.array(data)
        rs = 0
        if random_start:
            rs = random.Random().randint(0,len(data))
            kmeans = KMeans(n_clusters=k, random_state=rs,n_init=20).fit(npdata)
        else:
            init  =[]
            per = int(len(data)/k)
            for i in range(len(data)):
                if len(init) == k:
                    break
                if i % per == 0:
                    init.append(data[i])
            init = np.array(init)
            kmeans = KMeans(n_clusters=k,init = init,n_init = 1, random_state=rs).fit(npdata)

        clusters = list(kmeans.labels_)
        # result = [[] for var in range(k)]
        # for i in range(len(data)):
        #     result[clusters[i]].append(data[i])
        return  clusters

class Summarizor():

    def __init__(self):
        self.name = "FisrtVersion"
        # self.vectorizer = Word2Vec(weighted=False)
        # self.vectorizer = VecSyntacticDependency()
        self.vectorizer = MyVec()
        self.cluster = Cluster()
        self.dist = Dist()
        self.disttype =Distance.EUD
        self.weight = [10, 1, 1, 1]
        self.count_index = 0
        self.info = self.name+", weight="+str(self.weight)+", vector="+self.vectorizer.name+", distance="+str(self.disttype)
        self.cluewords =set()
        self.load_clue_words()
        self.target_tag = ["n","v","m"]

    def load_clue_words(self,path= Dir.res+"/parameter/summarization_parameter/clue_words"):
        list1 = ftools.read_lines(path)
        for var in list1:
            self.cluewords.add(var.strip())

    '''
    calculate the coverage value list
    input : sens_vector:   a list vector of sentences 
            essay_vector:  the vector of essay
    output: coverage_list: a coverage value list of each sentence vector
    '''
    def coverage_values(self,sens_vector,essay_vector):
        coverage_list =[]
        for sv in sens_vector:
            coverage_list.append(self.dist.sim(sv,essay_vector))
        return coverage_list

    '''
    calculate the relative value matrix
    input : sens_vector:     a vector list of sentences
    output: relative_matrix: a relative value matrix of each two sentences
    '''
    def relative_values(self,sens_vector):
        relative_matrix=[[0]*len(sens_vector) for var in range(len(sens_vector))]
        for i in range(len(sens_vector)-1):
            for j in range(i+1,len(sens_vector)):
                relative_matrix[i][j] = self.dist.sim(sens_vector[i],sens_vector[j])
                relative_matrix[j][i] = relative_matrix[i][j]
        return relative_matrix

    '''
    calcluate the clue value of each sentences
    input : sens_words: a words list of each sentence
    output: clues_list: a clues value list of each sentences 
    '''
    def clueswords_values(self,sens_words):
        clue_list=  [0]*len(sens_words)
        for i in range(len(sens_words)):
            sen_w = sens_words[i]
            for w in sen_w:
                if w in self.cluewords:
                    clue_list[i]=1
                    break
        return clue_list

    '''
    calcluate the entities list of each sentences(not duplicated)
    input : sens_words:    a words list of each sentences
            sen_tag   :    a words'tag list of each sentences
    output: entities_list: a entities words list of each sentences
    '''
    def entities_values(self,sens_words,sen_tag):
        entities_list =[]
        for i in range(len(sens_words)):
            tmp = []
            for j in range(len(sens_words[i])):
                if sen_tag[i][j][0] in self.target_tag[0]:
                    tmp.append(sens_words[i][j])
            entities_list.append(tmp)
        return entities_list

    '''
    calculate the score of input option
    input: option         : the setences index list of option
           coverage_list  : the coverage value list of each sentences
           relative_matrix: the relative value betweent each two sentences
           clues_list     : the clues value list of each sentences
           entities_list  : the entities words list of each sentences
    output: score         : the score of input option
    '''
    def score_option(self,option,coverage_list,relative_matrix,clues_list,entities_list):
        coverage_value,relative_value,clues_value,entities_value =0,0,0,0
        tmp = set()
        for i in range(len(option)):
            coverage_value+= coverage_list[option[i]]
            for j in range(len(option)):
                if i!= j:
                    relative_value+= relative_matrix[option[i]][option[j]]
            clues_value += clues_list[option[i]]
            for var in entities_list[option[i]]:
                tmp.add(var)
        entities_value+=len(tmp)

        score = self.weight[0] * coverage_value + self.weight[1] * relative_value \
                + self.weight[2] * clues_value + self.weight[3] * entities_value
        return score

    '''
    generate all possible options
    input : len: the length of sentences
            num: the length of abstract
    output: all possible options of abstract
    '''
    def generate_options(self, len, num,start =0):
        res = []
        if num == 1:
            for var in range(start, len):
                res.append([var])
            return res
        for i in range(start, len - num + 1):
            for var in self.generate_options( len, num - 1,i + 1):
                res.append([i] + var)
        return res

    '''
    seperate text into sentences and seperate and tag each sentences
    input : text: the original text
    output: sens      : the sentences of text
            sens_words: the words of each sentences
            sens_tag  : the words's tag of each setences
    '''
    def analyze(self,text):
        sens_words,sens_tag = [],[]
        sens = tools.seperate_sentences(text)
        for sen in sens:
            tmp_words,tmp_tag = [],[]
            for w,t in tools.sen_pog(sen):
                tmp_words.append(w)
                tmp_tag.append(t)
            sens_words.append(tmp_words)
            sens_tag.append(tmp_tag)
        return sens,sens_words,sens_tag

    def save_value(self,path,text,coverage_list,relative_matrix,clues_list,entities_list):
        ftools.check_filename(path)
        save_dict = {}
        save_dict['#$#'.join(text)] =[coverage_list,relative_matrix,clues_list,entities_list]
        tools.save_object(save_dict,path)

    def summarize(self,text,num =3):
        sens,sens_words,sens_tag = self.analyze(text)
        start = time.time()
        sen_vector,essay_vector = self.vectorizer.vectorize(sens_words,sens_tag)
        end = time.time()
        # print(end-start,"vector")
        # for eee in sen_vector:
        #     print(eee)
        #
        # print(essay_vector)
        self.count_index+=1
        coverage_list = self.coverage_values(sen_vector,essay_vector)
        relative_matrix = self.relative_values(sen_vector)
        clues_list = self.clueswords_values(sens_words)
        entities_list = self.entities_values(sens_words,sens_tag)
        options = self.generate_options(len(sens),num)
        max_value, best_option= 0,None
        tmp = []
        self.save_value(Dir.res + "/FirstVersion/file"+str(self.count_index),text,coverage_list,relative_matrix,clues_list,entities_list)
        for option in options:
            option_score = self.score_option(option,coverage_list,relative_matrix,clues_list,entities_list)

            # tmp.append(str(option_score))
            if option_score > max_value:
                best_option = option
                max_value = option_score
        abstract = [sens[var] for var in best_option]
        # print('\n'.join(tmp),max_value)
        return abstract

def load_file(filepath):
    def filter(sen):
        return sen.strip()

    tmp = ftools.read_lines(filepath)
    return "。".join(map(filter, tmp))

if __name__=="__main__":
    # test_file = Dir.res+"/cleandata_604/news/training_4.txt"
    # text = ftools.read_lines(test_file)
    name = "training_4.txt"
    text_path = Dir.res + "/cleandata_test/news/" + name
    text = load_file(text_path)
    summ = Summarizor()
    print(summ.info)
    res = summ.summarize(text)
    for line in res:
        print(line)
