import jieba
from numpy import *
import math
import Dir
import src.tools.FileTools as tools
from src.ResultProcess import ResultPropress as RP

class ROUGE:

    def get_common_string(self,array,b,lhs,i,j):
        if i ==0 or j ==0:
            return
        if b[i][j] ==2:
            self.get_common_string(array,b,lhs,i-1,j-1)
            array.append(lhs[i-1])
        elif b[i][j] ==1:
            self.get_common_string(array,b,lhs,i-1,j)
        else:
            self.get_common_string(array,b,lhs,i,j-1)

    def get_max(self,matrix):
        max_value =0
        for i in range(matrix.__len__()):
            for j in range(matrix[i].__len__()):
                if matrix[i][j] > max_value:
                    max_value = matrix[i][j]
        return int(max_value)

    # def __count_ngram(self,list1,list2):
    #     commoncount=0
    #     for word in list1:
    #         if word in list2:
    #             commoncount+=1
    #     return commoncount

    def seperate_words(self,sentence,chinese = True):
        if chinese :
            return  list(jieba.cut(sentence))
        else:
            return sentence.split(" ")

    ### 输入 abstract: list [ sentence1, sentece2, ... , sentencen]
    def createNgram(self,abstract,n,chinese = True):
        # print(abstract)
        result ={}
        words,count = [],0
        for sentence in abstract:
            # print('sentence',sentence)
            words.extend(self.seperate_words(sentence,chinese))
        # print(n)
        for i in range(words.__len__()-n+1):
            ### [i+1,..,i+n-1]
            gram = words[i]
            for j in range(i+1,i+n):
                gram+="-"+words[j]
            if gram not in result.keys():
                result[gram]=1
            else:
                result[gram]+=1
            count += 1
        result["allcount"] = count
        # for key in result.keys():
        #     print(key,result[key])
        return result

    def createSkipNgram(self, abstract, n, chinese=True,unigram = True):
        # print(abstract)
        result = {}
        words, count = [], 0
        for sentence in abstract:
            words.extend(self.seperate_words(sentence, chinese))
        if unigram:
            for word in words:
                if word not in result.keys():
                    result[word]= 1
                else:
                    result[word]+=1
                count+=1
        for i in range(words.__len__()):
            gram = words[i]
            for j in range(i+1, i + n+2):
                if j >= words.__len__():
                    break
                tmp = gram+ "-" + words[j]
                # print(i,j)
                if tmp not in result.keys():
                    result[tmp] = 1
                else:
                    result[tmp] += 1
                count += 1
        result["allcount"] = count
        # for key in result:
        #     print(key,result[key])
        return result

    def ngramScore(self,model_grams,standard_grams):
        hit= 0
        for gram in standard_grams.keys():
            if gram!= "allcount":
                h = 0
                if gram in model_grams.keys():
                    h = model_grams[gram] if model_grams[gram] <= standard_grams[gram] else standard_grams[gram]
                    hit +=h
        if standard_grams["allcount"] != 0:
            score = hit / standard_grams["allcount"]
        else :
            score = 0
        return model_grams["allcount"] , standard_grams["allcount"], hit,score

    ## 输入的摘要：list[sentence1,sentence2,....]
    ## 输入N的值
    ## 输出 rouge-n的值
    def rouge_n(self,abstract,standard_abstract,n=1,chinese= True):
        # print(type(abstract))
        if isinstance(abstract,str):
            abstract = abstract.strip()
            abstract = abstract.split("\n")
        if isinstance(standard_abstract,str):
            standard_abstract = standard_abstract.strip()
            standard_abstract = standard_abstract.split("\n")

        model_grams = self.createNgram(abstract,n,chinese)
        standard_grams = self.createNgram(standard_abstract,n,chinese)
        result = self.ngramScore(model_grams,standard_grams)
        return result

    def rouge_skip(self,abstract,standard_abstract,n=4,chinese= True):
        model_grams = self.createSkipNgram(abstract, n, chinese)
        standard_grams = self.createSkipNgram(standard_abstract, n, chinese)
        result = self.ngramScore(model_grams, standard_grams)
        return result

    def rouge_l(self,abstract,standard_abstract,n=1,chinese= True):
        model_grams = self.createSkipNgram(abstract, n, chinese)
        standard_grams = self.createSkipNgram(standard_abstract, n, chinese)
        result = self.ngramScore(model_grams, standard_grams)
        return result

    def compute_result(self,result_list,model ="A",alpha = 0.95):
        gramScoreBest = 0
        totalhit, totalGramCountP, totalGramCount = 0, 0, 0
        for result in result_list:
            if model == "A":
                totalhit += result[2]
                totalGramCount += result[1]
                totalGramCountP += result[0]
            elif model == "B":
                if result[3] > gramScoreBest:
                    gramScoreBest = result[3]
                    totalhit = result[2]
                    totalGramCount = result[1]
                    totalGramCountP = result[0]
            else:
                totalhit += result[2]
                totalGramCount += result[1]
                totalGramCountP += result[0]
        gramScore, gramScoreP, gramScoref = 0, 0, 0
        if totalGramCount != 0:
            gramScore = totalhit / totalGramCount
        else:
            gramScore = 0
        if totalGramCountP != 0:
            gramScoreP = totalhit / totalGramCountP
        else:
            gramScoreP = 0
        if (1 - alpha) * gramScoreP + alpha * gramScore > 0:
            gramScoref = (gramScoreP * gramScore) / ((1 - alpha) * gramScoreP + alpha * gramScore)
        else:
            gramScoref = 0
        return totalhit, totalGramCount, totalGramCountP, gramScore, gramScoreP, gramScoref

    ### 生成摘要数量需和标准摘要数量一致
    ### model :A (平均得分)，B（最好得分）
    ### func:   rouge_n: ngram   rouge_skip : skip ngarm rouge_l : lcs
    def compute_rouge(self,abstract_list,stand_abstract_list,n=1,model = "A",alpha = 0.5,chinese = True,funcT = rouge_n):
        result_list = []
        if abstract_list.__len__() != stand_abstract_list.__len__():
            return None
        else:
            for i in range(abstract_list.__len__()):
                result_list.append(funcT(self,abstract_list[i],stand_abstract_list[i],n=n,chinese = chinese))
            return self.compute_result(result_list,model,alpha)

    ## 输入的摘要：list[sentence1,sentence2,....]
    ## b = 准确率和召回率的因子
    ## 公式：F= （1+b^2）r*p/(r+b^2*p)
    ## 输出rouge-l的值
    def rouge_l(self,abstract,standard_abstract,b=-1):
        lcs,n,m,abstaabstact_count_complete =0,0,0,False
        for i in range(standard_abstract.__len__()):
            words=set()
            standard_abstract_sentence = standard_abstract[i]
            standard_abstract_words = self.seperate_words(standard_abstract_sentence)
            for j in range(abstract.__len__()):
                abstract_sentence = abstract[j]
                abstract_words = self.seperate_words(abstract_sentence)
                inter_words = self.longest_common_subsequence(abstract_words, standard_abstract_words)
                words = words | set(inter_words)
                if not abstaabstact_count_complete:
                    n += abstract_words.__len__()
            abstaabstact_count_complete = True
            lcs += words.__len__()
            m += standard_abstract_words.__len__()
        # print(lcs,n,m)
        p = lcs / n
        r = lcs / m
        # print(p,r)
        f = (1 + b * b) * p * r / (r + b * b * p)
        if b == -1:
            f = r
        return lcs,n,f

    def de_func(self,w):
        return math.sqrt(w)

    ## ABANDED
    ## 输入的摘要：list[sentence1,sentence2,....]
    ## b = 准确率和召回率的因子
    ## 公式：F= （1+b^2）r*p/(r+b^2*p)
    ## 输出rouge-w的值
    # def rouge_w(self,abstract,standard_abstract,b =-1,function  = func,de_funcrion = de_func()):
    #     lcs, n, m, abstaabstact_count_complete = 0, 0, 0, False
    #     for i in range(standard_abstract.__len__()):
    #         words = set()
    #         standard_abstract_sentence = standard_abstract[i]
    #         standard_abstract_words = self.seperate_words(standard_abstract_sentence)
    #         for j in range(abstract.__len__()):
    #             abstract_sentence = abstract[j]
    #             abstract_words = self.seperate_words(abstract_sentence)
    #             words = words | set(self.longest_common_subsequence(abstract_words, standard_abstract_words))
    #             if not abstaabstact_count_complete:
    #                 n += abstract_words.__len__()
    #         abstaabstact_count_complete = True
    #         lcs += function(words.__len__())
    #         m += function(standard_abstract_words.__len__())
    #     # print(lcs,n,m)
    #     p = de_funcrion((lcs / n))
    #     r = de_funcrion(lcs / m)
    #     print(p, r)
    #     f = (1 + b * b) * p * r / (r + b * b * p)
    #     if b == -1:
    #         f = r
    #     return f


    ## 输入的摘要：list[sentence1,sentence2,....]
    ## b = 准确率和召回率的因子
    ## 公式：F= （1+b^2）r*p/(r+b^2*p)
    ## 输出rouge-w的值
    def rouge_s(self,abstract,standard_abstract,b =-1,max_skip =4,chinese= True):
        abstract_words,standard_abstract_words =[],{}
        for k in range(standard_abstract.__len__()):
            sentence = standard_abstract[k]
            words = rouge.seperate_words(sentence, chinese)
            standard_abstract_words[k]=[]
            for i in range(words.__len__()):
                tmp = ""
                for j in range(1, max_skip):
                    if i + j < words.__len__():
                        tmp = words[i] + "_" + words[i + j]
                        standard_abstract_words[k].append(tmp)
                    else:
                        break

        for sentence in abstract:
            words = rouge.seperate_words(sentence,chinese)
            for i in range(words.__len__()):
                tmp = ""
                for j in range(1, max_skip):
                    if i + j < words.__len__():
                        tmp = words[i] + "_" + words[i + j]
                        abstract_words.append(tmp)
                    else:
                        break

        count_match,reference_match =0,0
        for i in range(standard_abstract_words.__len__()):
            reference_match+=standard_abstract_words[i].__len__()
            for word in standard_abstract_words[i]:
                if word in abstract_words:
                    count_match+=1
        # print(count_match,n)
        return count_match,reference_match,count_match/reference_match


    def demo_getlcs(self):
        rouge = ROUGE()
        lhs = ["police", "killed", "ended", "the", "gunman"]
        rhs = ["police", "ended", "the", "gunman"]
        result = rouge.longest_common_subsequence(lhs, rhs)
        print(result)

    def demo_rouge_n(self):
        rouge = ROUGE()
        # standard_abstract,abstract = [],[]
        # path_1 = Dir.resource + "extradata\\test\\Guess_Summ_1.txt"
        # path_2 = Dir.resource + "extradata\\test\\Guess_Summ_2.txt"
        # path_11 = Dir.resource + "extradata\\test\\Ref_Summ_1_1.txt"
        # path_21 = Dir.resource + "extradata\\test\\Ref_Summ_2_1.txt"
        # guess1 = tools.read_lines(path_1)
        # guess2 = tools.read_lines(path_2)
        # ref_1 = tools.read_lines(path_11)
        # ref_2 = tools.read_lines(path_21)
        standard_abstract = [["man kill police","police man kill police"]]
        abstract = [["police police man kill police"]]
        result1 = rouge.compute_rouge(abstract, standard_abstract,n=1,chinese = False)
        print(result1)
        result2 = rouge.compute_rouge(abstract, standard_abstract, n=2,chinese = False)

        print(result2)

    def demo_rouge_l(self):
        standard_abstract = ["w1 w2 w3 w4 w5"]
        abstract =["w1 w2 w6 w7 w8",
                            "w1 w3 w8 w9 w5 w0"]
        resul = self.rouge_l(abstract,standard_abstract)
        print(resul)

    def demo_rouge_s(self):
        standard_abstract =["police killed the gunman"]
        abstract = ["police kill the gunman"]
        abstract1 = ["the gunman kill police"]
        abstract2 = ["the gunman police killed"]

        result = self.rouge_n(abstract, standard_abstract, 1)
        print("n = 1 ", result)
        result = self.rouge_n(abstract1, standard_abstract, 1)
        print("n = 1 ", result)
        result = self.rouge_n(abstract2, standard_abstract, 1)
        print("n = 1 ", result)

        result = self.rouge_n(abstract, standard_abstract, 2)
        print("n = 2 ", result)
        result = self.rouge_n(abstract1, standard_abstract, 2)
        print("n = 2 ", result)
        result = self.rouge_n(abstract2, standard_abstract, 2)
        print("n = 2 ", result)
        #
        result = self.rouge_l(abstract, standard_abstract)
        print("rouge_l", result)
        result = self.rouge_l(abstract1, standard_abstract)
        print("rouge_l", result)
        result = self.rouge_l(abstract2, standard_abstract)
        print("rouge_l", result)

        result = self.rouge_s(abstract, standard_abstract)
        print("rouge_s ", result)
        result = self.rouge_s(abstract1, standard_abstract)
        print("rouge_s ", result)
        result = self.rouge_s(abstract2, standard_abstract)
        print("rouge_s ", result)

    ### 输入： rouge_list = [ [rouge_recall, rouge_precision, rouge_f],...[rouge_recall, rouge_precision, rouge_f] ]
    ###
    def average(self,rouge_list,option=1):
        average_recall,average_precision,average_f =0,0,0
        for tmp in rouge_list:
            average_recall += tmp[0]
            average_precision += tmp[1]
            average_f += tmp[2]


    def eval(self,abstract_dir, standard_dir,n = [1,2]):
        guess_summary_list = RP.get_file_path(abstract_dir)
        ref_summ_list = RP.get_file_path_ref(standard_dir)
        # print(guess_summary_list)
        # print(ref_summ_list)
        assay_guess=[]
        assay_ref = []
        for i in range(guess_summary_list.__len__()):
           assay_guess.append(tools.read(guess_summary_list[i]))
        for i in range(ref_summ_list.__len__()):
            # for k  in range(ref_summ_list)
            tmp = [tools.read(ref_summ_list[i][k]) for k in range(ref_summ_list[i].__len__())]
            assay_ref.append(tmp)
        # print(assay_ref.__len__())
        result =[]
        for r_n in n:
            recall_list =[]
            for i in range(assay_guess.__len__()):
                recall_list.append(sum([round(self.rouge_n(assay_guess[i],assay_ref[i][j],n= r_n,chinese=False)[-1],5) for j in range(assay_ref[i].__len__())]))
            value = sum(recall_list),recall_list.__len__(),sum(recall_list)/recall_list.__len__()
            result.append(value[-1])
        # print(sum(recall_list)/recall_list.__len__())
        return str(result)




rouge  = ROUGE()
guess_ = []
ref_guess = []


# rouge.demo_rouge_n()
# rouge.demo_rouge_s()
# s1 = list("abcbdab")
# s2 = list("bdcaba")
# rouge.longest_common_subsequence(s1,s2)

# abstract = ["police killed the gunman"]
# max_skip = 4
# abstract_words =[]
