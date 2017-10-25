import Dir
import re
from src.tools import FileTools as ftools
from src.tools import Tools as tools


### news: a list of sentences : list
### abstract: a list of sentences : list
def data_filter(news,abstract):
    if len(news)<10 or len(news) >=80:
        return -2
    if len(abstract)< 2 or len(''.join(abstract))<50 :
        return -1

    news_gram2,abstract_gram2 = [],[]
    for i in range(len(news)):
        tmp = tools.seperate(news[i])
        news_gram2.append(set([tmp[k]+tmp[k+1] for k in range(len(tmp)-1)]))
        # news_gram2.append(set(tmp))
    result = 0
    for i in range(len(abstract)):
        tmp = tools.seperate(abstract[i])
        abstract_gram2.append(set([tmp[k]+tmp[k+1] for k in range(len(tmp)-1)]))
        # abstract_gram2.append(set(tmp))
        value = 0
        for  j  in range(len(news_gram2)):
            v = len(abstract_gram2[i].intersection(news_gram2[j]))
            if v > value:
                value = v
        result+= value
    # print(news_gram2[12])
    # print(abstract_gram2[0])
    # print(abstract_gram2[0].intersection(news_gram2[12]))
    result /= sum([len(abstract_gram2[i]) for i in range(len(abstract_gram2))])
    # print(result)
    return result


def craw_result_process(root  = Dir.res+"/data/"):
    files = ftools.get_files(root)
    data = []
    for i in range(len(files)):
        filename = files[i]
        if len(data)>10:
            break
        lines = ftools.read_lines(root+filename)
        for line in lines:
            tmp = line.split(",")
            # print("news",len(tmp[2]))
            # print("news",tmp[2])
            #
            # print("abstract",len(tmp[1]))
            # print("abstract",tmp[1])

            abstract = tools.seperate_sentences(tmp[1])
            news = tools.seperate_sentences(tmp[2])
            print(abstract)
            print(news)
            # input()
            jude = data_filter(news,abstract)
            if jude >0.5:
                data.append(['\n'.join(abstract),'\n'.join(news)])
    return data

def save_data(data,save_root):
    news_root =save_root+"/news/"
    abst_root =save_root+"/abstract/"
    for i in range(len(data)):
        fname = "trainning_"+str(i)+".txt"

        ftools.write(abst_root+fname,data[i][0])
        ftools.write(news_root+fname,data[i][1])


if __name__ == "__main__":
    data = craw_result_process()
    save_data(data,save_root=Dir.res+"/cleandata_uncheck/")