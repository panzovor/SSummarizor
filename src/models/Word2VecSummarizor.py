from src.models import  Preprocess
# from gensim.models.doc2vec import Doc2Vec,LabeledSentence
from gensim.models.word2vec import Word2Vec
import numpy as np
import networkx as nx
from src.tools import FileTools as tools
import jieba
import re
import os
import Dir
# import smart_open

class Doc2VecSummarizor():

    def __init__(self):
        self.name="Doc2VecSummarizor"
        self.model = None
        self.corpus = []
        self.train_file = Dir.res + "/WikiCorpus/wiki.jian.txt"
        self.model = self.train()

    def wiki_preprocess(self,save_path = Dir.res+"/WikiCorpus/wiki.jian.seperate.txt"):
        tmp_result =[]
        # save_path = Dir.res+"WikiCorpus/wiki.jian.seperate.txt"
        index =0
        with open(self.train_file,"r") as train_corpus:
            # print("read complete")
            index = 0
            for line in train_corpus:

                # print(line)
                # input()
                regex = "。。。。。。|？|。|！|；|\.\.\.\.\.\."
                sentences = re.split(regex,line)
                for sen in sentences:
                    words = list(jieba.cut(sen.strip()))
                new_line = ' '.join(words)
                tmp_result.append(new_line)
                if tmp_result.__len__() == 5000:
                    # print(index*5000 + 5000)
                    tools.write_list(save_path,tmp_result,mode="a")
                    index +=1
                    tmp_result = []
            # print(tmp_result.__len__())
            tools.write_list(save_path, tmp_result, mode="a")




    ## sentences in train_file should be seperated
    def train(self,dimension=200,iter = 10,trainfile = Dir.res + "WikiCorpus/wiki.jian.seperate.txt",load_model_if_exits = True):
        model_path = Dir.res+"/W2V/w2v_"+str(dimension)+".model"
        if os.path.exists(model_path) and load_model_if_exits:
            self.model = Word2Vec.load(model_path)
            return self.model
        tmp =tools.read_lines(trainfile)
        index =0
        for string in tmp:
            words = (string.split(" "))
            self.corpus.append(words)
            # print(words)
            # index+=1
            # print(index)
            # Doc2Vec()
        self.model = Word2Vec(self.corpus,size = dimension,iter=iter, min_count=5)
        path = Dir.res+"W2V/w2v_"+str(dimension)+".model"
        if not os.path.lexists(Dir.res+"W2V/"):
            os.makedirs(Dir.res+"W2V/")

        self.model.save(path)
        return self.model

    def tickOOV(self,words):
        # print(words)
        new_words = []
        for word in words:
            if word in self.model.vocab:
                new_words.append(word)
        return new_words

    def similarity(self,sentence1, sentence2):
        words1 = self.tickOOV(list(jieba.cut(sentence1)))
        words2 = self.tickOOV(list(jieba.cut(sentence2)))
        # print(words1.__len__(),words2.__len__())
        sim = self.model.n_similarity(words1,words2)
        if isinstance(sim,np.ndarray):
            sim = 0
        return sim


    def pageRank(self,essay,nx_parameter = {'alpha': 0.8},num = 3):
        if isinstance(essay,str):
            essay= essay.replace("\r\n","")
            essay= essay.replace("\u3000","")
            sentences = re.split("。|？|！|；", essay)
        else:
            sentences = essay
        graph_array = np.zeros((sentences.__len__(), sentences.__len__()))
        for x in range(graph_array.__len__()):
            for y in range(x, graph_array[x].__len__()):
                graph_array[x, y] = self.similarity(sentences[x], sentences[y])
                graph_array[y, x] = self.similarity(sentences[y], sentences[x])
                # print(graph_array[x,y])
        nx_graph = nx.from_numpy_matrix(graph_array)
        try:
            score = nx.pagerank(nx_graph, **nx_parameter)
            sorted_score = sorted(score.items(), key=lambda item: item[1], reverse=True)
            abstract = {}
            for index, score in sorted_score:
                # item = (index,sentences[index],score)
                if abstract.__len__() < num:
                    abstract[index] = [''.join(sentences[index].split(" ")), score]
                    # print(item)
            abstract = sorted(abstract.items(), key=lambda item: item[0], reverse=False)
        except:
            abstract = [sentences[:num]]
        return abstract

    def summarize(self,essay,num=3):
        tmp = self.pageRank(essay,num=num)
        result = [res[1][0] for res in tmp]
        return result


    def demo(self):
        import Dir


if __name__ == "__main__":
    # pass
    import math
    # docS = Doc2VecSummarizor()
    # print(docS.model.vocab.__len__())
    # res = docS.model.similarity("女士","男子")
    # print(res)
    # res = docS.model.similarity("女士","自动步枪")
    # print(res)
    # print(docS.model["豆芽"])
    # print(docS.model["自动步枪"])
    # res = docS.model.similarity("豆芽","自动步枪")
    # print(res)
    docs = Doc2VecSummarizor()
    # docs.wiki_preprocess()
    docs.train(dimension=200)
    print(docs.model.n_similarity(["男人", "女人"], ["声音", "生意"]))
    print(docs.model.n_similarity(["男人", "女人"], ["快乐","悲伤"]))
    print(docs.model.similarity("男人","女人"))
    print(docs.model.most_similar("男人"))
    print(docs.model.most_similar("战争"))
    print(docs.model.most_similar("悲伤"))
    print(docs.model.similarity("快乐","悲伤"))
