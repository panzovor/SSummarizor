import math
import re
import numpy as np
import networkx as nx
import jieba.posseg as posseg

class New_Graph():

    def __init__(self):
        self.name ="New_Graph"

    ### 事件元素（比如名词，动词）
    target_label=['n']

    ### input : essay string
    ### output: sentences list
    def seperate_sentences(self,essay):
        regex=  "。|？|！|；"
        sentences = re.split(regex,essay)
        return sentences

    ### input : a sentence -string
    ### output: (words(list), pog(list)) (list,list)
    def pog(self,sentence):
        words, tags=  [],[]
        for word, tag in posseg.cut(sentence):
            words.append(word)
            tags.append(tag)
        return words,tags

    ### input: sentences-(list)
    ### output: elements(事件元素列表){'w1':[1,2,..,n], 'w2':[4,3,..,n]},{'1':[w1,w2,..,wn], '2':[w2,w1,...,wn]} (dict,dict)
    def get_node_list(self,sentences):
        elements,sen_words={},{}
        for i in range(sentences.__len__()):
            sen = sentences[i]
            words,tag = self.pog(sen)
            for i in range(words.__len__()):
                if tag[i] in self.target_label:
                    if words[i] not in elements.keys():
                        elements[words[i]] = []
                    elements[words[i]].append(i)
            # print(words[i])
                    if sen not in sen_words.keys():
                        sen_words[sen] = []
                    sen_words[sen].append(words[i])
            if sen not in sen_words.keys():
                sen_words[sen]= []
        return elements,sen_words

    ### input : words of sentences1 , words of sentences2 list,list
    ### output: similarity between sentences1 and sentences2 float
    def get_similarity(self,word_list1, word_list2):
        """默认的用于计算两个句子相似度的函数。
        Keyword arguments:
        word_list1, word_list2  --  分别代表两个句子，都是由单词组成的列表
        """
        words = list(set(word_list1 + word_list2))
        vector1 = [float(word_list1.count(word)) for word in words]
        vector2 = [float(word_list2.count(word)) for word in words]

        vector3 = [vector1[x] * vector2[x] for x in range(len(vector1))]
        vector4 = [1 for num in vector3 if num > 0.]
        co_occur_num = sum(vector4)

        if abs(co_occur_num) <= 1e-12:
            return 0.

        denominator = math.log(float(len(word_list1))) + math.log(float(len(word_list2)))  # 分母

        if abs(denominator) < 1e-12:
            return 0.

    ### input : words;e.g: [[w1,w2,...,wn]...[w11,w12,w1n]] 2 deminsion matrix
    ### output: a similarity matrix 2 deminsion matrix
    def compute_similarity(self,sen_words):
        if sen_words.__len__() == 0:
            return None
        sim_matrix = [np.zeros[sen_words.__len__()] for i in range(sen_words.__len__())]
        for i in range(sen_words.__len__()):
            for j in range(i, sen_words.__len__()):
                tmp = self.get_similarity(sen_words[i],sen_words[j])
                sim_matrix[i][j] = tmp
                sim_matrix[j][i] = tmp
        return sim_matrix

    ### input: 2 list
    ### ouput: the connecty of two list
    def get_connecty_strong(self,list1,list2):
        if max(list1.__len__(),list2.__len__()) == 0:
            return 0
        counter = 0
        for ele in list1:
            if ele in list2:
                counter+=1
        connecty = counter/(max(list1.__len__(),list2.__len__()))
        return connecty

    ### input:  elements(事件元素列表){'w1':[1,2,..,n], 'w2':[4,3,..,n]},{'1':[w1,w2,..,wn], '2':[w2,w1,...,wn]} (dict,dict)
    ### output: graph: 2 dimension array(list)
    def build_graph(self,elements_graph):
        elements = sorted(list(elements_graph.keys()))
        graph = np.zeros((elements.__len__(),elements.__len__()))

        for i in range(elements.__len__()):
            for j in range(i,elements.__len__()):
                if i == j :
                    graph[i][j] = 1
                else:
                    connecty = self.get_connecty_strong(elements_graph[elements[i]], elements_graph[elements[j]])
                    graph[i][j]= connecty
        return elements, graph

    ### input:  2 dimension array(list)
    ### output: index and it's importance (dict)
    def textRank(self,graph_array,nx_parameter = {'alpha': 0.8}):
        nx_graph = nx.from_numpy_matrix(graph_array)
        score = nx.pagerank(nx_graph, **nx_parameter)
        # sorted_score = sorted(score.items(), key=lambda item: item[1], reverse=True)
        return score

    def summarize(self,essay,num=4):

        sentences = []
        if not isinstance(essay, list):
            regex = "！|？|。|；|\.\.\.\.\.\."
            new_sentences = re.split(regex, essay)
            # print(sentences.__len__())
            for sen in new_sentences:
                if sen.strip().__len__() > 3:
                    sentences.append(sen.strip())
        else:
            # print("wrong  ")
            sentences = essay

        if sentences.__len__()<= num:
            return sentences
        elements_graph,sentence_elements = self.get_node_list(sentences)
        elements,graph = self.build_graph(elements_graph)
        elements_index= {}
        sentence_order={}
        for i in range(sentences.__len__()):
            sentence_order[sentences[i]] = i
        for i in range(elements.__len__()):
            if elements[i] not in elements_index.keys():
                # print(elements[i])
                elements_index[elements[i]] = i
        score = self.textRank(graph)
        # print(score)
        result = {}
        for sen in sentences:
            sen_elements = sentence_elements[sen]
            # print(sen_elements)
            importances = [score[elements_index[sen_ele]]  for sen_ele in sen_elements]
            result[sen] = sum(importances)
        score_sorted_abstract = sorted(result.items(), key= lambda d:d[1],reverse = True)[:num]
        selected_sentences_order ={}
        for sentence,score in score_sorted_abstract:
            selected_sentences_order[sentence] = sentence_order[sentence]
        # print(selected_sentences_order.__len__())
        order_sorted_abstract = sorted(selected_sentences_order.items(), key= lambda d:d[1])
        return [var for var, order in order_sorted_abstract]

    def demo(self):
        filepath = "/home/czb/project/Summarizor/resource/data/news.sentences/training1.txt"
        with open(filepath,mode="r") as file:
            essay = file.read()
        print(essay)
        abstract = self.summarize(essay)
        for sen in abstract:
            print(sen.strip())
        print(abstract)

# demo()

if __name__ =="__main__":
    ga= New_Graph()
    ga.demo()
