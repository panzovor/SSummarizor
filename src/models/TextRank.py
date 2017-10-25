import networkx as nx
import numpy as np
import jieba
import re
import math
import src.models.OptionSelector as OSelector
import src.tools.Tools as tools

'''
    参考文章引用：TextRank:Bringing Order into Texts
    参考文章链接：http://www.aclweb.org/old_anthology/W/W04/W04-3252.pdf

'''

class TextRank():


    def __init__(self):
        self.stop_words = set()
        self.name = "TextRank"

    def load_stop_words(self,path):
        with open(path,mode="r") as file:
            for line in file.readlines():
                if line.__len__()>0:
                    self.stop_words.append(line)

    def overlap_simi(self,words1,words2):
        overlap = 0
        for word in words1:
            if word in words2:
                overlap += 1
        similarity = overlap / (words1.__len__()+words2.__len__())
        return similarity

    def eucl_dist(self,v1,v2):
        vector1, vector2 = np.mat(v1), np.mat(v2)
        dist = np.sqrt(np.sum(np.square(vector1 - vector2)))
        return dist

    def cos_dist(self,v1,v2):
        vector1, vector2 = np.mat(v1), np.mat(v2)
        dist = np.sqrt(np.sum(np.square(vector1 - vector2))) / (np.sqrt(np.sum(np.square(vector1))) + np.sqrt(np.sum(np.square(vector2))))
        return dist

    def get_similarity(self,sen1, sen2):
        """默认的用于计算两个句子相似度的函数。
        Keyword arguments:
        word_list1, word_list2  --  分别代表两个句子，都是由单词组成的列表
        """
        word_list1 = tools.seperate(sen1)
        word_list2 = tools.seperate(sen2)

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

        return co_occur_num / denominator

    def vector_simi(self,words1,words2):
        whole_words = [word for word in words1]
        whole_words.extend(words2)
        # print(words1.__len__(),words2.__len__())
        vector1,vector2=np.zeros(whole_words.__len__()),np.zeros(whole_words.__len__())
        for i in range(whole_words.__len__()):
            if whole_words[i] in words1:
                vector1[i] = 1
            if whole_words[i] in words2:
                vector2[i] = 1
        # print("v1",vector1)
        # print("v2",vector2)
        # print("simi",func(self,vector1,vector2))
        return self.cos_dist(vector1,vector2)

    def similarity(self,sentence1,sentence2,del_stop_words=False):
        words1 = list(jieba.cut(sentence1))
        words2 = list(jieba.cut(sentence2))
        words_1,words_2 = words1,words2
        if del_stop_words:
            words_1 = [word for word in words1 if word not in self.stop_words]
            words_2 = [word for word in words2 if word not in self.stop_words]
        return self.vector_simi(words_1,words_2)


    ### input:  an essay ,no need to seperate sentence or words
    ### parameter: 'similarity':overlap_simi,vector_simi
    ### output: an dict of abstract, each item as a tuple( index, sentence, socre)
    def textRank(self,essay,config = {'similarity':vector_simi,'alpha': 0.85},sent_nums = 3):
        sentences = []
        if not isinstance(essay,list):
            regex = "！|？|。|；|\.\.\.\.\.\."
            new_sentences = re.split(regex,essay)
            # print(sentences.__len__())
            for sen in new_sentences:
                if sen.strip().__len__()>3:
                    sentences.append(sen.strip())
        else:
            # print("wrong  ")
            sentences = essay
        graph_array = np.zeros((sentences.__len__(),sentences.__len__()))

        nx_parameter = {'alpha': 0.8}
        if 'alpha' in config.keys():
            nx_parameter['alpha'] = config['alpha']
        simi_parameter = self.vector_simi
        if 'vector_simi' in config.keys():
            simi_parameter = self.vector_simi

        for x in range(graph_array.__len__()):
            for y in range(x,graph_array[x].__len__()):
                graph_array[x,y] = self.get_similarity(sentences[x],sentences[y])
                graph_array[y,x] =  self.get_similarity(sentences[y],sentences[x])
        nx_graph = nx.from_numpy_matrix(graph_array)
        score = nx.pagerank(nx_graph,**nx_parameter)
        sorted_score = sorted(score.items(),key = lambda item:item[1],reverse = True)
        abstract = {}
        for index,score in sorted_score:
            # item = (index,sentences[index],score)
            if abstract.__len__()< sent_nums:
                abstract[index] = [sentences[index],score]
                # print(item)
        abstract = sorted(abstract.items(), key=lambda item: item[0], reverse=False)
        index =0
        # for abst in abstract:
        #     print(index,abst)
        #     index+=1
        return abstract

    # def get_reasonable_abstract(self,sorted_score,graph,num):
    #     hou_num = num*2 if num*2 < graph.__len__() else graph.__len__()
        # answer = []
        # for i in range(hou_num):

    def summarize(self,essay,num=3,optionmital = False):
        if not optionmital:
            result = self.textRank(essay, sent_nums=num)
            result_abstract = [res[1][0] for res in result]
        else:
            result = self.textRank(essay,sent_nums= num*2)
            tmp = [res[1][0] for res in result]
            import_list = [res[1][1] for res in result]
            if num > tmp.__len__():
                num = tmp.__len__()
            data = [list(jieba.cut(line)) for line in tmp]
            options = OSelector.options_optimization(data,import_list,num)
            tmp_res = []
            for i in range(options[0].__len__()):
                tmp_res.append(sum(options[1][i]))
            max_index = tmp_res.index(max(tmp_res))
            result_abstract = [result[var][1][0] for var in options[0][max_index]]

        return result_abstract

def demo():
    TR = TextRank()
    content =u"元旦小长假高速不免费。市民出行前应注意天气状况 如遇恶劣天气高速极有可能封路。还有几天就要到元旦了，随着元旦小长假的临近，一方面市民的聚会活动不断增加，另外商场打折促销活动也开始增多，路面交通压力也将随之加大。市交管部门昨天发布元旦期间交通情况预报，预测12月31日晚间，工体、中华世纪坛、蓝港附近等区域车多人多，极易出现交通拥堵情况。交管部门还提醒，元旦期间高速公路不免通行费，如市民自驾出行，需留意天气情况，避免出行受到影响。交管部门表示，从近3年元旦假期交通运行数据看，市区和高速公路交通压力均呈现逐年上升态势。预计2017年元旦假期，市区交通压力仍将不断上升。高速方面，从日均流量看，2014年、2015年、2016年分别为107.3万辆、130.3万辆、146.2万辆。交管部门预计，2017年元旦高速公路总体流量将较往年有进一步增加，特别是京藏、机场、京港澳、京开、京承5条高速公路交通流量较为集中。交管部门表示，假期如遇恶劣天气，市民应尽量选择其他交通方式出行。交管部门联合高德等通过数据分析，提醒市民下列区域容易拥堵。受大型活动影响，12月31日夜间工人体育场、首都体育馆、人民大会堂、国家大剧院、北京音乐厅、国图音乐厅、中华世纪坛、三里屯、蓝色港湾、世贸天阶、太庙、奥林匹克森林公园周边区域交通压力较大。近期城区西部的西单商圈，城区东部的朝阳北路大悦城、大望桥区、国贸桥区、三里屯、工体周边；城区北部的积水潭、五道口、四通桥区、清河五彩城，城区南部的首地大峡谷等地区容易拥堵。另外，各滑雪场以及密云古北水镇、房山金陵旅游景区、门头沟戒台寺等景点易堵。通往滑雪场、景点的京藏、京承、大广、京昆、京平等高速公路交通压力将可能增大。交管部门称，针对年底道路交通压力明显的特点，各交通支大队专门制定疏导方案，采取多项措施，加强疏导。在启动区域高等级上勤方案的基础上，围绕环路、主干道、高速公路、联络线等主要道路和重点餐饮、购物娱乐场所周边道路全面加强管控。交管部门透露，近期将连续组织开展全市夜查集中统一行动，重点围绕商场、酒吧、饭店、影剧院周边等活动集中、人员车辆密集场所加大严查、打击力度。"
    para = {'similarity':TR.overlap_simi,'alpha': 0.85}
    # print(para)
    # result = TR.textRank(content)
    # print(result.__len__())
    result = TR.summarize(content,4)
    for res in result:
        print(res)



if __name__ =='__main__':
    demo()