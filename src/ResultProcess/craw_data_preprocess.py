from src.tools import Tools as tls
import Dir
import os
import shutil
from src.tools import FileTools as tools

''' 对爬取的结果进行筛选，主要筛选出抽取式的句子（组成文摘的句子均来自于文章（或具有足够高的相似度》0.9））  '''
def check_if_extract(abstract,content,sim_alpha = 0.9,num_alpha=0.9):
    regex = "。|\?|\!|？|！"
    content_lines = re.split(regex, content)[:-1]
    abstract_lines = re.split(regex, abstract)[:-1]
    count = 0
    result = []
    if abstract_lines.__len__()<=0:
        return False,[]
    for abstract_line in abstract_lines:
        for content_line in content_lines:
            # print(content_line)
            # print(abstract_line)
            try:
                common_string = longest_common_subsequence(content_line,abstract_line)
            # print(common_string)
                if common_string.__len__()/abstract_line.__len__() >sim_alpha:
                    count+=1
                    result.append(content_lines.index(content_line))
                    break
            except:
                continue
    result.append(abstract_lines)
    result.append(content_lines)
    # input()
    if count/abstract_lines.__len__() >num_alpha:
        return True,result
    else:
        return False,[]

def longest_common_subsequence( lhs, rhs):
    # print(lhs.__len__(),rhs.__len__(),rhs)
    m, n = lhs.__len__() + 1, rhs.__len__() + 1
    c = [[0 for col in range(n)] for row in range(m)]
    b = [[0 for col in range(n)] for row in range(m)]
    for i in range(1, m):
        for j in range(1, n):
            if lhs[i - 1] == rhs[j - 1]:
                c[i][j] = c[i - 1][j - 1] + 1
                b[i][j] = 2
            elif c[i - 1][j] >= c[i][j - 1]:
                c[i][j] = c[i - 1][j]
                b[i][j] = 1
            else:
                c[i][j] = c[i][j - 1]
                b[i][j] = -1
    result = []
    get_common_string(result, b, lhs, lhs.__len__(), rhs.__len__())
    # print(result)
    return result

def get_common_string(array, b, lhs, i, j):
    if i == 0 or j == 0:
        return
    if b[i][j] == 2:
        get_common_string(array, b, lhs, i - 1, j - 1)
        array.append(lhs[i - 1])
    elif b[i][j] == 1:
        get_common_string(array, b, lhs, i - 1, j)
    else:
        get_common_string(array, b, lhs, i, j - 1)

def filter(name):
    if "info" in name:
        return True
    return False

def check_extract(file_dir,save_path):
    files = []
    tools.get_filelist(file_dir,files,filter)
    extract_result = set()
    un_first_result =set()
    analysis_result ={}
    for file in files:
        # print(file)
        content = tools.read(file)
        content = re.sub("\[|\]|", "", content)
        lines = content.split("\n")
        for line in lines:
            tmp = line.split("', '")
            if tmp.__len__() ==3:
                extract = check_if_extract(tmp[1],tmp[2])
                if extract[0]:
                    extract_result.add(line)
                    if tmp[0] not in analysis_result.keys():
                        analysis_result[tmp[0]] = []
                    analysis_result[tmp[0]] = extract[1]

                    all_value = sum(extract[1][:-2])
                    supose_value = 0
                    low_ = get_sum(extract[1][-2])
                    hight_ = get_sum(extract[1][-1])
                    # print(tmp[0], all_value, low_, hight_, extract[1][:-2],extract[1][-2:])
                    # print(extract_result.__len__())
                    if all_value> low_+2:
                        # print(tmp[0], all_value, low_, hight_, extract[1][:-2], extract[1][-2:])
                        un_first_result.add(line)

                    print(extract_result.__len__(),un_first_result.__len__())

                else:
                    pass
            else:
                # print("format error",tmp.__len__())
                # print(line)
                pass
        # print("exract",extract_result.__len__())
    tools.write_list(save_path,extract_result)
    tools.write_list(save_path+".txt", un_first_result)

def get_sum(index):
    value = 0
    for i in range(index):
        value+=i
    return value

# file_dir = Dir.resource+"\data_extract"
# save_path = Dir.resource+"\extract_data_process\data_processed_9.9.txt"
# check_extract(file_dir,save_path)
#
# content_line = "比如朱日和基地和飞弹，习先生表示，有关部署不是针对台湾"
# abstract_line ="比如朱日和基地和飞弹，习先生表示，有关部署不是针对台湾"
# result = longest_common_subsequence(content_line,abstract_line)
# print(result.__len__())
import re
def seperate_sentences(essay):
    regex = "。。。。。。|？|！|；|\.\.\.\.\.\.|。"
    tmp = re.split(regex,essay)
    result =[]
    for tm in tmp:
        if tm.__len__()>4:
            result.append(tm)
    return result

def generate_data(file=Dir.res+"/extract_data_process/data_processed_9.9.txt",savePath=Dir.res+"/extract_data_process/data"):
    content = tools.read_lines(file)[1:-1]
    data = {}
    for file in content:
        file = file.replace("&nbsp;","")
        tmp = str(file[1:-1]).split("', '")
        if tmp[1] not in data.keys():
            data[tmp[1]] = tmp[2]
    index = 0
    for key in sorted(data.keys()):
        save_content = savePath+"/news/training_"+str(index)+".txt"
        save_abstract = savePath+"/abstract/training_"+str(index)+".txt"
        tools.write_list(save_content,seperate_sentences(data[key]))
        tools.write_list(save_abstract,seperate_sentences(key))
        index+=1

def lcs_fast(left,right):
    left_dict,right_dict = {},{}
    for var in left:
        if var not in left_dict.keys():
            left_dict[var] =0
        left_dict[var]+=1
    for var in right:
        if var not in right_dict.keys():
            right_dict[var] = 0
        right_dict[var]+=1
    count =0
    for var in left_dict.keys():
        if var in right_dict.keys():
            count+= min(left_dict[var],right_dict[var])
    return count


def get_abstract_index(news,abstracts):
    matrix =[[0]*len(news) for var in range(len(abstracts))]
    res =[]
    for k in range(len(abstracts)):
        # print(abstracts[i][k])
        for j in range(len(news)):
            matrix[k][j] = lcs_fast(news[j], abstracts[k])
        # print(matrix[k].index(max(matrix[k])),news[i][matrix[k].index(max(matrix[k]))])
        max_index = matrix[k].index(max(matrix[k]))
        if matrix[k][max_index]>0.5:
            res.append(max_index)
    res.sort()
    return res

def filter_craw_data(data_dir = Dir.res+"/craw_data/data/",save_dir= Dir.res+"/cleandata_none"):
    if os.path.lexists(save_dir):
        shutil.rmtree(save_dir)

    files = tools.get_files(data_dir)
    cleandata =[]
    count = 0
    bad_sample =[]
    for i in range(len(files)):
        print(i,len(files),len(cleandata))
        fname = files[i]
        path = data_dir+fname
        lines = tools.read_lines(path)
        for line in lines:
            line = line.strip()

            # try:
            if 1:
                last_ = line.rindex(",")
                first_ = line.index(",")
                if first_==last_:
                    continue
                tmp = [line[:first_],line[first_+1:last_],line[last_+1:]]
                abstracts = tls.seperate_sentences(tmp[1])
                news = tls.seperate_sentences(tmp[2])

                tmp = get_abstract_index(news,abstracts)

                count += 1
                if len(tmp) != len(abstracts):
                    continue
                # print(tmp)
                # cmd = input()
                # if "1" in cmd:
                #     print('\n'.join(abstracts))
                #     print("--------------------")
                #     print('\n'.join(news))
                #
                #     print("--------------------")
                #     print("words:",w_count)
                w_count = 0
                for li in news:
                    w_count += len(tls.seperate(li))
                if w_count < 520:
                    continue

                if sum(tmp[:3]) <=3:
                    continue
                cleandata.append([abstracts,news])
                tools.write(save_dir+"/abstract/trainning_"+str(len(cleandata))+".txt",'\n'.join(abstracts))
                tools.write(save_dir+"/news/trainning_"+str(len(cleandata))+".txt",'\n'.join(news))
            # except Exception as e:
            #     print(str(e),e.with_traceback(e.__traceback__))
            #     print("error",line)
            #     bad_sample.append(line)
    print(count,len(bad_sample),len(cleandata))

def clean(data_dir = Dir.res+"/cleandata_8700/news/"):
    flist= tools.get_files(data_dir)
    # print(data_dir,len(flist))
    for fname in flist:
        flag = False
        content = tools.read(data_dir+fname)
        if "3805" in fname:
            print(content)
            input()

        if "您的浏览器不支持video标签\n" in content:
            content = content.replace("您的浏览器不支持video标签\n","")
            flag = True
        if "新闻 专题 微博" in content :
            flag = True
            content = content[:content.index("新闻 专题 微博")]

        if flag:
            print(fname)
            tools.write(data_dir+fname, content)

### craw_data
### total 29491
### good sample 8700
# def generate_data_sen_not_all_form():
if __name__ == "__main__":
    # file=Dir.res+"/extract_data_process/data_processed_9.9.txt.txt"
    # savePath=Dir.res+"/extradata/"
    # print(file,savePath)
    # generate_data(file,savePath)

    # print(lcs_fast("aaaa","bbbbaaa"))
    # filter_craw_data()
    clean()