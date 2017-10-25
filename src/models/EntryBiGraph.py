import src.models.HITS as HITS
import jieba.posseg as posg
import re
import src.tools.Tools as tools
'''
   将每个句子中的名词看作二部图的一边的节点，将每个句子看作二部图的另一边的节点，若句子seni中出现名词wordj则连接seni 与 wordj
   将得到的二部图转化未
   根据得到的图使用hits算法，计算节点的authority 以及 hub值，取节点的authority值作为该节点的重要性。
   根据句子节点所对应的authority重要性来对句子进行排序，选择重要前n个句子作为最后的摘要。
   参考文章引用：Parveen D, Strube M. Integrating Importance, Non-Redundancy and Coherence in Graph-Based Extractive Summarization[C]//IJCAI. 2015: 1298-1304.
   参考文章链接 ：http://michael.kimstrube.de/papers/parveen15.pdf
'''
class EntryBiGraph():
    def __init__(self):
        self.name ="EntryBigraph"


    def build_graph(self,sentences):
            # sentences = essay
        # print(sentences)
        entry_graph,sent_graph = {},{}
        for i  in range(sentences.__len__()):
            sen = sentences[i]
            sent_graph[i] = set()
            for word,tag in posg.cut(sen):
                # print(word,tag)
                if str(tag)  == "n":
                    sent_graph[i].add(word)
                    if word not in entry_graph.keys():
                        entry_graph[word] = set()
                    entry_graph[word].add(i)
        # print_graph(entry_graph)
        # print_graph(sent_graph)
        return entry_graph,sent_graph

    def generate_normal_graph(self,sent_graph):
        normal_graph ={}
        nodes = sorted(list(sent_graph.keys()))
        for i in range(nodes.__len__()-1):
            for j in range(i+1,nodes.__len__()):
                if sent_graph[nodes[i]].intersection(sent_graph[nodes[j]]).__len__()>0:
                    if nodes[i] not in normal_graph.keys():
                        normal_graph[nodes[i]] = []
                    normal_graph[nodes[i]].append(nodes[j])
        # print(normal_graph.__len__())
        return normal_graph

    def print_graph(self,normal_graph):
        for key in normal_graph.keys():
            print(key,normal_graph[key])

    def summarize(self,essay,num=3):
        sentences = tools.seperate_sentences(essay)
        if sentences.__len__() <= num:
            return sentences
        # print(sentences.__len__())
        mid_graph = self.build_graph(sentences)
        graph = self.generate_normal_graph(mid_graph[1])
        # print_graph(graph)
        if graph.__len__()==0:
            return sentences[:num]
        au,hub = HITS.HITS(graph)
        sorted_au = sorted(au.items(), key = lambda item:item[1], reverse = True)
        sorted_hub = sorted(hub.items(), key = lambda item:item[1], reverse = True)
        result = []
        for res in sorted_au[:num]:
            # print(res)
            result.append(int(res[0]))
        result.sort()
        abstract = []
        for res in result:
            abstract.append(sentences[res])
        # for sent in abstract:
        #     print(sent)
        return abstract

    def optimization(self,au,mid_graph,graph,abstract):
        importance, coherrence, redundancy = 0,0,0
        for sent in abstract:
            importance+=au[sent]
            coherrence+=graph[sent].__len__()/(sent+1)
            redundancy += mid_graph[0][sent].__len__()
        redundancy = abstract.__len__()*redundancy/mid_graph[1].__len__()
        return sum([importance,coherrence,redundancy])




    def demo(self):
        # file = Dir.resource+"data/news.sentences/training1.txt"
        file = "/home/czb/PycharmProjects/Summarizor/resource/extract_data_process/data1000/abstract/training_1070.txt"
        with open(file,"r") as data:
            essay = data.readlines()
        self.summarize(essay)


# demo()