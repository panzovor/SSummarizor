__author__ = 'E440'
import math
import numpy as np

### result,[0,0,...,0,0],int,int
def options_generator_fast(result,value,all_sentence_num,selected_sentences_num,index=-1):
    selected_sentences_num -=1
    if selected_sentences_num == -1:
        return False
    tmp = [var for var in value]
    # print(selected_sentences_num,"======",index+1,all_sentence_num-selected_sentences_num,"======")
    for i in range(index+1,all_sentence_num-selected_sentences_num):
        if tmp[i] == 0:
            tmp[i] =1
            flag = options_generator_fast(result,tmp,all_sentence_num,selected_sentences_num,index=i)
            tmp1 = [var for var in tmp]

            if not flag:
                tmp_res = []
                for k in range(tmp1.__len__()):
                    if tmp1[k]>0:
                        tmp_res.append(k)
                result.append(tmp_res)
                # print(tmp1)
                # result.append(tmp_res)
            tmp[i] =0
    return True

def option_generator(all_sentence_num,selected_sentences_num):
    data = np.zeros(all_sentence_num)
    result =[]
    options_generator_fast(result,data,all_sentence_num,selected_sentences_num)
    return result

def vect_sim_cons(vector1,vector2):
    con = sum([vector1[i]* vector2[i] for i in range(vector1.__len__())])
    vec1_len = math.sqrt(sum([math.pow(var,2) for var in vector1 ]))
    vec2_len = math.sqrt(sum([math.pow(var,2) for var in vector2 ]))
    return con/(vec1_len*vec2_len)

def sim_overlap(sentence1,sentence2):
    # print(sentence1,sentence2)
    whole_words = list(set(sentence1).union(set(sentence2)))
    vector1,vector2 = np.zeros(whole_words.__len__()),np.zeros(whole_words.__len__())
    for word in sentence1:
        vector1[whole_words.index(word)]+=1
    for word in sentence2:
        vector2[whole_words.index(word)]+=1
    return vect_sim_cons(vector1,vector2)

def sim_matrix(data):
    matrix = [np.zeros(data.__len__()) for i in range(data.__len__())]
    for i in range(data.__len__()):
        for j in range(i,data.__len__()):
            matrix[i][j] = sim_overlap(data[i],data[j])
            matrix[j][i] = matrix[i][j]
    return matrix

def options_analysis_nonredundancy(options,sim_matrix):
    analysis_result = []
    for option in options:
        tmp=[]
        for i in range(option.__len__()-1):
            for j in range(i+1,option.__len__()):
                tmp.append(sim_matrix[option[i]][option[j]])
        analysis_result.append(1-sum(tmp)/tmp.__len__())

    return analysis_result

def options_analysis_importance(options,imp_list):
    result =[]
    for option in options:
        # value =0
        # for var in option:
        #     value+= imp_list[var]
        # print("tmp",[imp_list[var] for var in option])
        result.append(sum([imp_list[var] for var in option]))
    return result

## sentence1 and sentence2 should be sepearte by words
## sentence1 before sentence2
def flu_calculator(sentence1,sentence2):
    value_list = []
    for i in range(sentence1.__len__()):
        word  = sentence1[i]
        # print(word,sentence2)
        if word in sentence2:
            value_list.append(sentence1.__len__()-i+sentence2.index(word))
    if value_list.__len__() == 0:
        return 0
    return 1-(sum(value_list)/value_list.__len__())/\
             (sentence1.__len__()+sentence2.__len__())

def fluen_matrix(data):
    # print(data)
    matrix = [np.zeros(data.__len__()) for i in range(data.__len__())]
    for i in range(data.__len__()-1):
        for j in range(i+1,data.__len__()):
            # print(data[i],data[j])
            matrix[i][j] = flu_calculator(data[i],data[j])
    return matrix

def options_analysis_fluence(options,fluen_matrix):
    result = []
    for option in options:
        tmp = []
        for i in range(option.__len__()-1):
            for j in range(i+1,option.__len__()):
                # print(option[i],option[j],fluen_matrix[option[i]][option[j]])
                tmp.append(fluen_matrix[option[i]][option[j]])
        result.append(sum(tmp)/tmp.__len__())
    return result

def options_optimization(data,import_list,seleceted_num):
    analysis_result=[]
    options = option_generator(data.__len__(),seleceted_num)
    # print("options_len",options.__len__())
    s_matrix = sim_matrix(data)
    redundancy = options_analysis_nonredundancy(options,s_matrix)
    f_matrix = fluen_matrix(data)
    fluence = options_analysis_fluence(options,f_matrix)
    importance = options_analysis_importance(options,import_list)
    # print(importance)
    for i in range(options.__len__()):
        analysis_result.append([redundancy[i],fluence[i],importance[i]])
        # print(options[i],analysis_result[i])
    return options,analysis_result

# data = ['sentence1','sentence2','sentence3','sentence4','see']
# selected_num = 2
# options_optimization(data,selected_num)

# data=['i love the pai','your dont love the pai','pai is your favor']
# data=[['i','love','the','pai'],
#       ['your','dont','love','the','pai'],
#       ['pai','is','your','favor']]
#
# result = options_optimization(data,2)
# print('options','redundancy','fluency')
# for res in result:
#     print(res)

def demo():
    file = "E:/new 3.txt"
    text = open(file,mode='r',encoding='utf-8').read()
    print(text)
    import re
    sentences = re.split("。|？|！|；",text)
    import jieba
    data = []
    for sentence in sentences:
        tmp = list(jieba.cut(sentence))
        if tmp.__len__()>0:
            data.append(tmp)
        # print(list(segmentation.cut(sentence)))
    print(data.__len__())
    result = options_optimization(data,3)
    final_result = []
    for i in range(result[0].__len__()):
        print(result[1][i])
        final_result.append(sum(result[1][i]))
        # for index in result[0][i]:
        #     print(data[index],end="")
        # print()
    o_index = final_result.index(max(final_result))
    print(o_index)
    print(result[0][o_index])
    for var in result[0][o_index]:
        print(data[var])

if __name__ =="__main__":
    # print("ddd")
    demo()