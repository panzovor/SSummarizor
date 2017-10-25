import re
import src.models.OptionSelector as OptionSelector
import numpy as np
import Dir
import math
import jieba


class FirstIdea():
    def __init__(self):
        self.name = "First_idea"

    ### input : essay string
    ### output: sentences list
    def seperate_sentences(self, essay):
        regex = "。|？|！|；|\n"
        # print(essay)
        sentences = re.split(regex, essay)
        result = []
        for sen in sentences:
            if sen.strip().__len__()>6:
                # print("ddd",sen.strip())
                result.append(sen.strip())
        return result

    ### input : words of sentences1 , words of sentences2 list,list
    ### output: similarity between sentences1 and sentences2 float
    def get_similarity(self, word_list1, word_list2):
        """ 默认的用于计算两个句子相似度的函数。
            Keyword arguments:
            word_list1, word_list2  --  分别代表两个句子，都是由单词组成的列表
        """
        words = list(set(word_list1 + word_list2))
        vector1 = [float(word_list1.count(word)) for word in words]
        vector2 = [float(word_list2.count(word)) for word in words]
        vector3 = [vector1[x] * vector2[x] for x in range(len(vector1))]
        vector4 = [1 for num in vector3 if num > 0.]
        co_occur_num = sum(vector4)
        # print(co_occur_num)
        if abs(co_occur_num) <= 1e-12:
            return 0.

        denominator = math.log(float(len(word_list1))) + math.log(float(len(word_list2)))
        # print(denominator)
        if abs(denominator) < 1e-12:
            return 0.

    def get_simi_matrix(self, sentences):
        simi_matrix = [np.zeros(sentences.__len__()) for i in range(sentences.__len__())]
        for i in range(sentences.__len__()):
            # print(sentences[i])
            for j in range(i, sentences.__len__()):
                if i == j:
                    simi_matrix[i][j] = 1
            else:
                words1 = list(jieba.cut(sentences[i]))
                words2 = list(jieba.cut(sentences[j]))
                # print(words1)
                # print(words2)
                simi_matrix[i][j] = self.get_similarity(words1, words2)
                # print(simi_matrix[i][j])
                simi_matrix[j][i] = simi_matrix[i][j]
        return simi_matrix

    def get_sim_two_sentences(self,sentence1,sentence2):
        words1 = list(jieba.cut(sentence1))
        words2 = list(jieba.cut(sentence2))
        # print(words1)
        # print(words2)
        return self.get_similarity(words1, words2)

    def summarize(self,sentences,num =4, lr = 0.05,threshold = 0.2):
        abstract_cluster =[]
        if num >1:
            seed = []
            simi_matrix = self.get_simi_matrix(sentences)
            row_min = min(simi_matrix)
            row = simi_matrix.index(row_min)
            collum = row_min.index(min(row_min))
            seed.append(row,collum)
            abstract_cluster.append([row])
            abstract_cluster.append([collum])
            while seed.__len__()< num:
                tmp = np.zeros(sentences.__len__())
                for i in range(seed.__len__()):
                    tmp += np.array(simi_matrix[seed[i]])
                index = tmp.__index__(min(tmp))
                seed.append(index)
                abstract_cluster.append([index])
        else:
            seed = [0]
            abstract_cluster.append([0])

        # for i in range(sentences.__len__()):
        #     tmp = []
        #     for j in range()





    def summarize(self, sentences, num=4):
        # sentences = self.seperate_sentences(content)
        # for sen in sentences:
        #     print(sen)
        abstract = []
        simi_matrix = self.get_simi_matrix(sentences)
        sen_len = sentences.__len__()
        tmp = []
        for i in range(sen_len - 1):
            tmp.append(simi_matrix[i][i + 1])
        chafen = []
        for i in range(tmp.__len__() - 1):
            chafen.append(tmp[i + 1] - tmp[i])
        seperate_id = [0]
        # print(chafen)
        for i in range(num):
            index = chafen.index(max(chafen))
            seperate_id.append(index)
            chafen[index] = -1.0
        seperate_id = sorted(seperate_id)
        # print(seperate_id)
        i = 0
        while i< seperate_id.__len__()-1:
            # print(i)
            if seperate_id[i]!= seperate_id[i+1]:
                abstract.append(sentences[seperate_id[i]:seperate_id[i + 1]])
            else:
                abstract.append([sentences[seperate_id[i]]])
                i+=1
            i+=1
        # print(abstract.__len__(),num)
        if abstract.__len__()< num:
            abstract.append([sentences[seperate_id[-1]]])
        result = []
        all_num = sum([var.__len__() for var in abstract])
        # if all_num>num:
        #     print("ddddddddddddddddddddddddddddd")
        for line in abstract:
            result.append(line[0])
        # OptionSelector.op
        return result


    def demo(self):
        filepath = Dir.resource+"data/news.sentences/training1.txt"
        with open(filepath, mode="r") as file:
            essay = file.read()
        # print(essay)
        abstract = self.summarize(essay)
        for sen in abstract:
            print(sen)

# fd = FirstIdea()
# fd.demo()