from src.tools import Tools as tools
from src.tools import FileTools as ftools
from src.ResultProcess import CrawResult_process as crpss
import os
import Dir

def filter(path = Dir.res+"/extradata/"):
    # print(os.path.abspath(path))
    news_path = path+"news/"
    abstract_path = path+"abstract/"

    news_file_list = os.listdir(news_path)
    abst_file_list = os.listdir(abstract_path)

    bad_sample = []
    news = []
    for name in news_file_list:
        # if name in bad_sample:
        #     continue
        news.append(ftools.read_lines(news_path+name))
    abstracts =[]
    for name in abst_file_list:
        # if name in bad_sample:
        #     continue
        abstracts.append(ftools.read_lines(abstract_path+name))


    res =[]
    res_sen = []
    for i in range(len(news)):
        # print(news_file_list[i], abst_file_list[i], True if news_file_list[i] == abst_file_list[i] else False)
        matrix = [[0 for var in range(len(news[i]))] for var in range(len(abstracts[i]))]
        tmp =[]
        tmp_sen = []
        try:
            for k in range(len(abstracts[i])):
                # print(abstracts[i][k])
                for j in range(len(news[i])):
                    matrix[k][j] = len(crpss.longest_common_subsequence(news[i][j],abstracts[i][k]))
                # print(matrix[k].index(max(matrix[k])),news[i][matrix[k].index(max(matrix[k]))])
                max_index = matrix[k].index(max(matrix[k]))
                tmp.append(max_index)
                tmp_sen.append(news[i][max_index])
            # print(len(tmp),True if len(tmp) == len(abstracts[i]) else False)
        except:
            bad_sample.append(news_file_list[i])
            # print(news_file_list[i])

        res.append([news_file_list[i]]+tmp)
        res_sen.append([news_file_list[i]]+ tmp_sen)
    # for bb in bad_sample:
    #     print(bb)


    #     res.append(tmp)
    # print(bad_sample)
    # for i in range(len(res)):
    #     tmp = res[i]
    #     print(news_file_list[i],tmp,len(news[i]),len(abstracts[i]) , True if len(abstracts[i] ) == len(tmp) else False)
    return res,res_sen

def clean(path = Dir.res+"/extradata/",save = Dir.res+"/cleandata_1073/"):
    res = filter(path)
    clean_data = []
    for tmp in res:
        # print(tmp)
        if 0 not in tmp or 1 not in tmp:
            clean_data.append(tmp)
    for cd in clean_data:
        if cd[0] == "training_288.txt":
            print("skip------------------")
            continue
        print(cd[0])
        news_path = save+"news/"+cd[0]
        abstract_path =  save+"abstract/"+cd[0]
        ftools.copy(path+"news/"+cd[0],news_path)
        ftools.copy(path+"abstract/"+cd[0], abstract_path)

def get_clue_words(path = Dir.res+"/extradata/",savepath = Dir.res+"/parameter/summarization_parameter/clue_words", word_index = 3):
    _, res_sen = filter(path)
    words = {}
    for  var in res_sen:
        for  sen in var[1:]:
            ws = tools.seperate(sen)
            for w in ws[:word_index]:
                if w not in words.keys():
                    words[w] =0
                words[w]+=1

    content = ""
    for w in words.keys():
        content+= w+","+str(words[w])+"\n"
    ftools.write(savepath+str(word_index),content)





if __name__ == "__main__":
    # clean()
    for i in range(1,4):
        print(i)
        get_clue_words(word_index=i)

