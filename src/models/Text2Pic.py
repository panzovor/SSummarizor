from src.tools import FileTools as ftools
from src.tools import Tools as tools
import matplotlib.pyplot as plt
import Dir

def text2pic(text):
    sens = tools.seperate_sentences(text)
    nodes = []

    nodes_dict = {}
    sen_words =[]
    sen_noun_words =[]
    for sen in sens:
        wp = tools.sen_pog(sen)
        tmp_w =[]
        tmp_p = []
        tmp_noun =[]
        for w,p in wp:
            if "n" in p or "v" in p or "m" in p:
                if w not in nodes:
                    nodes.append(w)
                if w not in nodes_dict.keys():
                    nodes_dict[w] = 0
                nodes_dict[w]+=1
                tmp_noun.append(w)
            # tmp.append([w,p])
            tmp_w.append(w)
            tmp_p.append(p)
        sen_noun_words.append(tmp_noun)
        sen_words .append([tmp_w,tmp_p])
    # nodes = []
    # tmp = sorted(nodes_dict.items(), key= lambda d:d[1],reverse=True)
    # for var,count in tmp:
    #     nodes.append(var)
    #
    # print(tmp)

    matrix = [[0]*len(nodes) for var in range(len(nodes))]
    for k in range(len(sen_noun_words)):
        var = sen_noun_words[k]
        for i in range(len(var)-1):
            for j in range(i+1,len(var)):
                #
                matrix[nodes.index(var[i])][nodes.index(var[j])] += 1
                matrix[nodes.index(var[j])][nodes.index(var[i])] += 1
                # nouni_index = sen_words[k][0].index(var[i])
                # nounj_index = sen_words[k][0].index(var[j])
                # if nouni_index == nounj_index-1 and True:
                #     matrix[nodes.index(var[i])][nodes.index(var[j])] +=1
                #     matrix[nodes.index(var[j])][nodes.index(var[i])] +=1
                # else:
                # for p in sen_words[k][1][nouni_index:nounj_index]:
                #     if "v" in p or "m" in p:
                #         matrix[nodes.index(var[i])][nodes.index(var[j])] += 1
                #         matrix[nodes.index(var[j])][nodes.index(var[i])] += 1
                #         break
    return matrix,nodes

def sum2pic(text,nodes):
    sens = tools.seperate_sentences(text)
    sen_n =[]
    sen_w =[]
    sen_p = []
    # nodes_dict = {}
    for sen in sens:
        wp = tools.sen_pog(sen)
        tmp_sen_n =[]
        tmp_sen_w = []
        tmp_sen_p =[]
        for w,p in wp:
            if  ("n" in p or "v" in p or "m" in p )and w in nodes:
                tmp_sen_n.append(w)
            # if w not in nodes_dict.keys():
            #     nodes_dict[w] = 0
            # nodes_dict[w] += 1
            tmp_sen_w.append(w)
            tmp_sen_p.append(p)
        sen_n.append(tmp_sen_n)
        sen_w.append(tmp_sen_w)
        sen_p.append(tmp_sen_p)

    # nodes = []
    # tmp = sorted(nodes_dict.items(), key=lambda d: d[1], reverse=True)
    # for var, count in tmp:
    #     nodes.append(var)

    # print(tmp)

    matrix = [[0]*len(nodes) for var in range(len(nodes))]
    for i in range(len(sen_n)):

        for j in range(len(sen_n[i])):
            for k in range(j+1,len(sen_n[i])):
                # nouni_index = sen_w[i].index(sen_n[i][j])
                # nounj_index = sen_w[i].index(sen_n[i][k])
                matrix[nodes.index(sen_n[i][j])][nodes.index(sen_n[i][k])] += 1
                matrix[nodes.index(sen_n[i][k])][nodes.index(sen_n[i][j])] += 1
                # if nouni_index == nounj_index-1 and True :
                #     matrix[nodes.index(sen_n[i][j])][nodes.index(sen_n[i][k])] +=1
                #     matrix[nodes.index(sen_n[i][k])][nodes.index(sen_n[i][j])] +=1
                # for p in sen_p[i][nouni_index:nounj_index+1]:
                #     if "v" in p or "m" in p:
                #         matrix[nodes.index(sen_n[i][j])][nodes.index(sen_n[i][k])] += 1
                #         matrix[nodes.index(sen_n[i][k])][nodes.index(sen_n[i][j])] += 1
                #         break
    return matrix

def show_matrix(matrix):
    for i in range(len(matrix)):
        print(" ".join(map(str,matrix[i])))

def normalize_pic(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            sum_matrixi = sum(matrix[i])
            matrix[i][j] = matrix[i][j]/sum_matrixi if sum_matrixi !=0 else 0

def load_file(filepath):
    def filter(sen):
        return sen.strip()
    tmp = ftools.read_lines(filepath)
    return "。".join(map(filter,tmp))



if __name__ =="__main__":
    # text = "近日，教育部官方网站发布“关于宣布失效一批规范性文件的通知”，该通知由教育部、国务院学位委员会和国家语委联合发布，通知宣布《关于继续实施“985工程”建设项目的意见》、《“985工程”建设管理办法》、《“211工程”建设实施管理办法》等一批规范性文件失效。此举再一次引发关于“废除985、211工程”的联想。教育部在通知中表示，这些规范文件不利于稳增长、促改革、调结构、惠民生，已进行专项清理。已失效的规范性文件不再作为行政管理的依据。同时，在此次教育部宣布失效的文件中，还包括了“关于继续实施“优势学科创新平台”建设的意见”，“关于实施“特色重点学科项目”的意见”和“关于加快推进世界一流大学和高水平大学建设的意见”等一批文件。一年前，网上曾传“国家废除985、211工程”的消息，教育部当时回应称不存在废除的情况。事实上，这两大工程早就不再增加新的高校，两大工程已经关上大门。近年来，官方正逐渐淡化985、211工程概念。2015年，国家启动双一流大学建设计划，这一计划是目前国家新一轮重点大学建设的指导文件。"
    # text_sum = "近日，教育部网站发布“关于宣布失效一批规范性文件的通知”，该通知宣布《关于继续实施985工程建设项目的意见》、《985工程建设管理办法》、《211工程建设实施管理办法》等一批规范性文件失效。官方正逐渐淡化985、211概念。"
    badsamples = [15,]
    name = "training_4.txt"
    text_path = Dir.res+"/cleandata_604/news/"+name
    sum_path = Dir.res+"/cleandata_604/abstract/"+name
    text = load_file(text_path)
    text_sum = load_file(sum_path)

    # print('\n'.join(tools.seperate_sentences(text)))
    # print("----------------------------")
    # print('\n'.join(tools.seperate_sentences(text_sum)))

    pic,nodes = text2pic(text)
    # normalize_pic(pic)
    pic_s = sum2pic(text_sum,nodes)
    # normalize_pic(pic_s)

    f, (a1,a2) = plt.subplots(1, 2,figsize = (10,5))
    a1.imshow(pic)
    a2.imshow(pic_s)
    f.show()
    # plt.draw()
    plt.waitforbuttonpress()
