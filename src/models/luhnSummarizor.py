import Dir
import re
import jieba
import src.tools.FileTools as tools
import src.tools.DataLoader as loader

class Summarizor_luhn:

#算法流程：
#1.	分词并去掉常用词，词干化处理
#2.	统计文章的词频
#3.	设定关键词的词频上下界（大于等于5），确定关键词
#4.	对文章的每个句子进行截取，截取第一个关键词到最后一个关键词内所包括的句子部分	S。
#5.	对截取后的结果S进行筛选，若S中被少于4个非关键词隔开，则将S加入候选集
#计算得分，对每个S计算得分，计算公式为
#score = all/(all-significant)
#all: 全部词，significant: 关键词
#6.	选取固定数量的句子组成摘要

    def __init__(self):
        self.commonwords =[]
        self.load_commonwords()

    ## 载入停用词,路径：E:\PythonWorkSpace\Summarizor\resource\extradata\models\commonwords
    def load_commonwords(self):
        # file= "E:\PythonWorkSpace\Summarizor\\resource\extradata\luhn\commonwords"
        file= Dir.resource+"/extradata/luhn/commonwords"
        self.commonwords = tools.read(file)



    ## 分句处理
    ## 输入：文章，输出：文章的每个句子，类型dict:[(0:sen1),(1:sen2),...]
    # def seperate_sentence(self,essay):
    #     result ={}
    #     i=0
    #     regex = "。|\?|\!|？|！"
    #     tmp = re.split(regex,essay)
    #     for sentence in tmp:
    #         result[i] = sentence
    #         i+=1
    #     return result

    ## 分词处理
    ## 输入：句子列表，输出：每个句子的分词结果，类型list:[[word1,word2...],[word1,word2...],...]
    def seperate_words_per_sentence(self,sentences):
        result ={}
        for index in sentences.keys():
            result[index] = (list(jieba.cut(sentences[index])))
        return result


    ## 统计词频
    ## 输入：每个句子的分词结果
    ## 输出：词频统计结果，类型为dict:(words,frequency)
    def word_frequency(self, data):
        words =[]
        result ={}
        for tmp in data.keys():
            words.extend(data[tmp])
        for word in words:
            if word not in result.keys():
                result[word] = 0
            result[word] += 1
        return result


    ## 获取关键词：去掉常用词以及词频低于指定阈值（默认大于等于5）的词语
    ## 输入：词频统计结果
    def get_keywords(self, data, threshold=5):
        for key in list(data.keys()):
            if int(data[key]) < 5 or key in self.commonwords:
                data.pop(key)
        # print(data.keys())
        # print(data.keys())
        return list(data.keys())


    ## 计算得分
    ## 输入：句子的分词 list，输出：句子的得分，0：表示该句不选入候选集
    def score(self,sentence,keywords):
        start,end,allwords,non_keywords =0,0,0,0
        for i in range(sentence.__len__()):
            if sentence[i] in keywords:
                start =i
                break
        for i in range(sentence.__len__()):
            if sentence[sentence.__len__() - i-1] in keywords:
                end = sentence.__len__() - i-1
                break
        allwords = end -start+1
        for i in range(start,end):
            if sentence[i] not in keywords:
                non_keywords+=1
        if non_keywords>4:
            return 0
        else:
            return allwords/(non_keywords+1)

    def score_coutkeywords(self,sentence,keywords):
        tmp = set(sentence).intersection(set(keywords))
        return tmp.__len__()/sentence.__len__()

    ## 获取得分靠前的n个句子索引
    ## 输入：句子索引及得分，类型dict：{(0,0),(1,2),...},n
    ##  总字数限制
    ## 输出：索引,类型 list:[3,5,..]

    def get_higher_sentences(self,sentences,sentence_scores,n,num =-1):
        tmp = sorted(sentence_scores.items(), key=lambda d: d[1], reverse=True)
        result,tmpStr =[],""
        for key in tmp:
            key = list(key)
            if result.__len__() <n:
                if sentences[key[0]].__len__()>10:
                    if num>0:
                        if tmpStr.__len__() <num:
                            result.append(key)
                            tmpStr+=sentences[key[0]]
                        else:
                            break;
                    else:
                        result.append(key)
            else:
                break
        result  = sorted(list(dict(result).keys()))
        return result

    def index_sentences(self,sentences):
        resul ={}
        for i in range(sentences.__len__()):
            if sentences[i].strip().__len__() >0:
                resul[i] = sentences[i]
        return resul

    ## 生成摘要
    ## 输入:文章句子
    ## 输出:文章摘要(list ,句子的集合)
    def summarize(self,essay,num = -1):
        sentences = self.index_sentences(essay)
        sentences_words = self.seperate_words_per_sentence(sentences)
        frequency = self.word_frequency(sentences_words)
        # print((sorted(frequency.items(), key=lambda d: d[1], reverse=True)))
        keywords = self.get_keywords(frequency)
        # print(keywords)
        sentences_socres = {}
        for index in sentences_words.keys():
            # sentences_socres[index] = self.score(sentences_words[index], keywords)
            sentences_socres[index] = self.score_coutkeywords(sentences_words[index], keywords,)
        # print(sentences_socres)
        indexs = self.get_higher_sentences(sentences,sentences_socres,5,num)
        # print(indexs)
        # for index in indexs:
        #     print(index,sentences_socres[index])
        # for t in tmp:
        #     print(t, sentences_socres[t], set(sentences_words[t]).intersection(set(keywords)),
        #           sentences_words[t].__len__())

        result =[]
        for index in indexs:
            # print(index, sentences_socres[index], set(sentences_words[index]).intersection(set(keywords)),
            #       sentences_words[index].__len__())
            result.append(sentences[index])
        return result


# def demo():
#     summarizor = Summarizor_luhn()
#     essay=tools.read_lines(Dir.resource+"extradata/models/training64.txt")
#     print(type(essay))
#     result = summarizor.summarize(essay=essay)
#     print("========================")
#     for line in result:
#         print(line)
#
#
# demo()








