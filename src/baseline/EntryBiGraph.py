import src.models.HITS as HITS
import jieba.posseg as posg
import re
import src.tools.Tools as tools
from src.models.FirstVersion import  TextRank
'''
   将每个句子中的名词看作二部图的一边的节点，将每个句子看作二部图的另一边的节点，若句子seni中出现名词wordj则连接seni 与 wordj
   将得到的二部图转化未
   根据得到的图使用hits算法，计算节点的authority 以及 hub值，取节点的authority值作为该节点的重要性。
   根据句子节点所对应的authority重要性来对句子进行排序，选择重要前n个句子作为最后的摘要。
   参考文章引用：Parveen D, Strube M. Integrating Importance, Non-Redundancy and Coherence in Graph-Based Extractive Summarization[C]//IJCAI. 2015: 1298-1304.
   参考文章链接 ：http://michael.kimstrube.de/papers/parveen15.pdf
   
   
   Topical Coherence for Graph-based Extractive Summarizationv?
'''

class EntryBiGraph():
    def __init__(self):
        self.name ="EntryBigraph"
        self.targets = ["n"]
        self.info = self.name+ "_"+"_".join(self.targets)

    def set(self,targets):
        if targets!=None:
            self.targets = targets
        self.info = self.name + "_targets:" + str(self.targets)

    ### max(sum(au[i])+sum(au[i])+sum(e[i]) i in options
    def optimization(self,au,od,e):
        max_val,max_option = 0,[0,1,2]
        for i in range(len(e.keys())):
            for j in range(len(e.keys())):
                for k in range(len(e.keys())):
                    if i!=j and j!=k and i!=k:
                        au_value = (au[i]+au[j]+au[k])
                        od_value = (od[i]+od[j]+od[k])
                        e_value = (e[i]+e[j]+e[k])

                        if au_value+od_value+e_value > max_val:
                            max_val = au_value+od_value+e_value
                            max_option=[i,j,k]
        return max_option


    def build_graph(self,sentences):
            # sentences = essay
        # print(sentences)
        entry_graph,sent_graph = {},{}
        for i  in range(sentences.__len__()):
            sen = sentences[i]
            sent_graph[i] = set()
            words, tags = tools.seperate_pog(sen)
            for word_i in range(len(words)):
                word = words[word_i]
                tag = tags[word_i]
                # print(word,tag)
                if str(tag) in self.targets or "all" in self.targets or ("n" in str(tag) and "all_n" in self.targets):
                    sent_graph[i].add(word)
                    if word not in entry_graph.keys():
                        entry_graph[word] = set()
                    entry_graph[word].add(i)
        # print_graph(entry_graph)
        # print_graph(sent_graph)
        return entry_graph,sent_graph

    def generate_bigraph(self,built_graph):
        egraph,sgraph = built_graph[0],built_graph[1]
        bigraph = {}
        for node in sgraph.keys():
            if node not in bigraph.keys():
                bigraph[node] = []
            bigraph[node].extend(sgraph[node])
        for node in egraph.keys():
            if node not in bigraph.keys():
                bigraph[node] = []
            bigraph[node].extend(egraph[node])
        return bigraph

    def generate_normal_graph(self,sent_graph):
        normal_graph ={}
        nodes = sorted(list(sent_graph.keys()))
        for i in range(nodes.__len__()-1):
            for j in range(i+1,nodes.__len__()):
                if nodes[i] not in normal_graph.keys():
                    normal_graph[nodes[i]] = []
                if sent_graph[nodes[i]].intersection(sent_graph[nodes[j]]).__len__()>0:
                    normal_graph[nodes[i]].append(nodes[j])
        if len(nodes)-1 not in normal_graph.keys():
            normal_graph[len(nodes)-1] = []
        # print(normal_graph.__len__())
        return normal_graph

    def print_graph(self,normal_graph):
        for key in normal_graph.keys():
            print(key,normal_graph[key])

    def summarize(self,essay,num=3,fname = None):
        sentences = tools.seperate_sentences(essay)
        if sentences.__len__() <= num:
            return sentences
        # print(sentences.__len__())
        mid_graph = self.build_graph(sentences)
        bigraph = self.generate_bigraph(mid_graph)
        graph = self.generate_normal_graph(mid_graph[1])
        # print_graph(graph)
        au,hub = HITS.HITS(bigraph)

        od = {}
        for node in graph.keys():
            od[node] = len(graph[node])/(node+1)
        e = {}
        for node in mid_graph[1].keys():
            e[node] = len(mid_graph[1][node])

        options = self.optimization(au,od,e)
        abstract = []
        for var in options:
            abstract.append(sentences[var])

        return abstract

    def demo(self):
        # file = Dir.resource+"data/news.sentences/training1.txt"
        file = "/home/czb/Project/Summarizor/resource/cleandata_604/news/training_4.txt"
        with open(file,"r") as data:
            essay = data.readlines()
        print(essay)
        res = self.summarize(essay)
        for line in res:
            print(line)

if __name__ == "__main__":

    import time
    start = time.time()
    print(start)
    eg = EntryBiGraph()

    eg.demo()
    end = time.time()
    print(end-start)