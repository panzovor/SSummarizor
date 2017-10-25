import codecs
import os
import re
import sys

def read(filename,encoding="utf-8"):
    file = codecs.open(filename,encoding=encoding)
    content = file.read()
    file.close()
    return content

def read_lines(filename,encoding ="utf-8"):
    file = codecs.open(filename, encoding=encoding)
    lines = file.readlines()
    content=[]
    for line in lines:
        line = line.strip()
        line = line.replace("\r\n","")
        if line.__len__()>0:
            content.append(line)
    file.close()
    return content

def read_dict(path,seperate=" "):
    res ={}
    with open(path,mode="r") as file:
        for line in file:
            line=  line.strip()
            if seperate in line:
                index = line.rindex(seperate)
                key = line[:index].strip()
                value = line[index:].strip()
                if key not in res.keys():
                    res[key] = value
    return res

def write(filename,content,encoding="utf-8",mode = 'w'):
    check_filename(filename)
    file = codecs.open(filename=filename,mode=mode,encoding=encoding)
    file.write(str(content))
    file.flush()
    file.close()
import shutil
def copy(src,dir_):
    check_filename(dir_)
    shutil.copy(src,dir_)

def write_list(filename,list_content,encoding = "utf-8",mode = "w"):
    content = '\n'.join(list_content)
    # for line in list_content:
    #     content+= line+"\n"
    write(filename,content,encoding,mode)

def print_proccess(i,a,p=10):
    if i%int(a/p) == 0:
        sys.stdout.write("*")
        sys.stdout.flush()
    if i == a-1:
        print()



def write_dict(filename,dict_content,encoding = "utf-8"):
    content  =""
    for key in dict_content.keys():
        content+= key+"\t"+str(dict_content[key])+"\n"
    write(filename,content,encoding)


def write_com_dict(filename,dict_content,encoding = "utf-8"):
    content  =""
    for key in dict_content.keys():
        content+= key+","+','.join(list(map(str,dict_content[key])))+"\n"
    write(filename,content,encoding)

def nothing(s):
    return

def seperate_sentence(essay):
    result = {}
    i = 0
    regex = "。|\?|\!|？|！"
    tmp = re.split(regex, essay)
    for sentence in tmp:
        result[i] = sentence
        i += 1
    return result


def del_file(file):
    os.remove(file)

def get_files(path):
    return os.listdir(path)

def get_filelist(dir, fileList,filter=nothing):
    newDir = dir
    if os.path.isfile(dir):
        fileList.append(dir)
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            #如果需要忽略某些文件夹，使用以下代码
            #if s == "xxx":
                #continue
            if filter(s):
                continue
            newDir=os.path.join(dir,s)
            get_filelist(newDir, fileList)
    return fileList

def check_filename(filename):
    if not os.path.exists(filename):

        path = filename[:filename.rfind("/")]
        if not os.path.exists(path):
            # print(path)
            os.makedirs(path)

def check_build_file(filename):
    if not os.path.exists(filename):
        with open(filename,mode="w"):
            pass


def isexists(path):
    if os.path.exists(path):
        return True
    return False

def read_dir_content(file_dir,filter = nothing):
    filelist=[]
    get_filelist(file_dir,filelist,filter)
    # print(filelist)
    result =[]
    for file in filelist:
        result.append(read(file))
    return result

def read_dir_lines(file_dir,filter = nothing):
    filelist=[]
    get_filelist(file_dir,filelist,filter)
    # print(filelist)
    result =[]
    for file in filelist:
        result.append(read_lines(file))
    return result

def get_name(filepath):
    start = filepath.rfind("/")
    end = filepath.rfind(".")
    if end ==0:
        end = filepath.__len__()
    name = filepath[start+1:end]
    return name

def replace_nothing(s):
    return s

def read_dir_lines_dict(file_dir,filter = nothing,replace_fun = replace_nothing):
    filelist = []
    get_filelist(file_dir,filelist,filter)
    result = {}
    for file in filelist:
        name = get_name(file)
        name = replace_fun(name)
        if name not in result.keys():
            result[name]=""
        data = read_lines(file)
        result[name] = data
    return result

if __name__ == "__main__":
    import Dir
    path = Dir.res+"/sen_data/604_sen_corpus.txt.txt"
    print(path)
    res = read_dict(path)
    print(len(res))
    # for key,value in res.items():
    #     print(key,value)