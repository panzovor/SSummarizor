import src.tools.FileTools as tools
import jieba
import os
import Dir
import shutil
'''将生成的文摘以及标准文摘转换为“数字+空格”的形式 进行保存，以便调用perl版本的rouge计算相应指标'''
###  将文件夹中的每个文件的内容进行切词，保存在指定文件夹中（该文件夹中不能有和原文件夹文件名字相同的文件否则会被覆盖）
def result_process(file_dir,save_dir):
    if os.path.lexists(save_dir):
        shutil.rmtree(save_dir)
    filenames = []
    tools.get_filelist(file_dir,filenames)
    for file in filenames:
        content  = tools.read_lines(file)
        name = tools.get_name(file)
        result =[]
        for line in content:
            words = jieba.cut(line)
            string = ""
            for word in words:
                string+= word+" "
            string = string[:-1]
            result.append(string)
            save_path = save_dir+"/"+name+".txt"
            tools.write_list(save_path,result)

def build_word_index(file_dir,words_path):
    filename = []
    def filter(s):
        if "all" in s:
            return True
        return False
    tools.get_filelist(file_dir, filename, filter)
    whole_words = {}
    for file in filename:
        lines = tools.read_lines(file)
        for line in lines:
            words = list(jieba.cut(line))
            for word in words:
                if word.__len__() > 0:
                    if word not in whole_words.keys():
                        whole_words[word] = whole_words.__len__()
    word_index = ""
    for word in whole_words.keys():
        word_index += word + ":" + str(whole_words[word]) + "\n"
    tools.write(words_path, word_index)
    return whole_words

### 读取分词后的文件，并将其中的词语用唯一的一个数值代替。
def replace_words_by_num(whole_words,file_dir,save_dir):
    if os.path.lexists(save_dir):
        shutil.rmtree(save_dir)
    filename = []
    def filter(s):
        if "all" in s:
            return True
        return False
    tools.get_filelist(file_dir,filename,filter)
    content = {}
    for file in filename:
        lines = tools.read_lines(file)
        string = ""
        for line in lines:
            words = line.split(" ")
            for word in words:
                if word.__len__()>0:
                    if word in whole_words.keys():
                        string+= str(whole_words[word])+" "
            string = string.strip()
            string+="\n"
        content[tools.get_name(file)] = string
        # print(string)
        # input()
    for name in content:
        savepath = save_dir+name+".txt"
        tools.write(savepath,content[name])

### 从指定文件夹中获取每个文件名（参考摘要的格式）
def get_dirfiles_into_list_abstract(file_dir,replace_dir):

    list,result  = [],{}
    tools.get_filelist(file_dir,list)
    for listfile in list:
        filename = tools.get_name(listfile)
        filename = filename[8:filename.__len__()-1]
        if filename not in result.keys():
            result[filename] = []
        if replace_dir =="":
            result[filename].append(listfile)
        else:
            result[filename].append(str(replace_dir+"/"+tools.get_name(listfile)+".txt"))
    return result

def numberize_all_data(corpus_dir,word_index_save,ref_dir,ref_dir_save,abstract_dir,abstract_dir_save):
    word_index = build_word_index(corpus_dir,word_index_save)
    replace_words_by_num(word_index,ref_dir,ref_dir_save)
    replace_words_by_num(word_index,abstract_dir,abstract_dir_save)


def get_file_path(file_dir):
    result=[]
    for file in os.listdir(file_dir):
        if "all" not in file:
            result.append(file_dir+file)
            # print(file)
    result.sort()
    return result

def get_file_path_ref(file_dir):
    result = []
    fex = ""
    for file in os.listdir(file_dir):
        # print(file,'ddddd')
        if "a.txt" in file:
            fex = "a.txt"
            result.append(file[:file.index("a.txt")])
            # print(file)
        elif "b.txt" not in file:
            result.append(file[:file.index(".txt")])
            fex = ".txt"
        # print('----------',result[-1])
        # else:
        #     result.append(file[:file.index("b.txt")])
    result.sort()
    for i in range(result.__len__()):
        result[i] =[file_dir+result[i]+fex]
    return result



### 从指定文件夹中获取每个文件名（生成摘要）
def get_dirfiles_into_list_luhn(file_dir,replace_dir):
    list,result  = [],{}
    tools.get_filelist(file_dir,list)
    for listfile in list:
        filename = tools.get_name(listfile)
        filename = filename[8:]
        if filename not in result.keys():
            result[filename ]= []
        if replace_dir == "":
            result[filename] = (listfile)
        else:
            result[filename].append(str(replace_dir + "/" + tools.get_name(listfile)+".txt"))
    return result

### 并生成rouge（PythonRouge）所需要的参数格式
### file_dir_standar: 参考摘要所在目录
### file_dir_peer: 生成的摘要所在目录
### replace_dir: ubuntu下的目录（ROUGE 在ubuntu下）,默认为该空（即原来所在的目录）
def make_parameter(file_dir_standard, file_dir_generated,replace_dir_standard="",replace_dir_generated = ""):
    result_abstract = get_dirfiles_into_list_abstract(file_dir_standard,replace_dir_standard)
    result_luhn = get_dirfiles_into_list_luhn(file_dir_generated,replace_dir_generated)
    resultString =""
    luhnString =""
    if set(result_abstract.keys()).intersection(set(result_luhn.keys())).__len__() == result_abstract.keys().__len__():
        for key in result_abstract.keys():
            tmp = str(result_abstract[key])
            resultString+= tmp+","
            luhnString+= str(result_luhn[key])[1:-1]+","

    luhnString = luhnString[:-1]
    resultString = resultString [:-1]
    luhnString = "["+luhnString+"]"
    resultString = "["+resultString+"]"
    print(luhnString)
    print(resultString)

def demo_make_parameter():
    file_dir_standard = Dir.resource + "data/abstract"
    file_dir_generated = Dir.resource + "result/luhn_result"
    replace_dir_standard = "Summarization_Result/2016-10-28/standard"
    replace_dir_generated = "Summarization_Result/2016-10-28/generated"
    make_parameter(file_dir_standard,file_dir_generated,replace_dir_standard,replace_dir_generated)


def demo_result_process():
    file_dir = Dir.resource+"result/luhn_result"
    save_dir = Dir.resource+"result/abstract_seperate"
    result_process(file_dir,save_dir)

# demo_result_process()
# corpus_dir = Dir.resource+"data/news.sentences"
# word_index_save = Dir.resource+"result/word_index/"
#
# ref_dir = Dir.resource+"result/abstract_seperate"
# ref_dir_save = Dir.resource+"result/luhn_result_process_r/"
#
# abstract_dir = Dir.resource+"result/abstract_result"
# abstract_dir_save = Dir.resource+"result/abstract_result_r/"
# numberize_all_data(corpus_dir,word_index_save,ref_dir,ref_dir_save,abstract_dir,abstract_dir_save)

#
# filepath = '/home/czb/PycharmProjects/Summarizor/resource/result/tr_result/ref_processed/training1b.txt'
# import src.tools.FileTools as tools
# content = tools.read(filepath)
# print(content)
