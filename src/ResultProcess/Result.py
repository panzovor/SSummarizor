import Dir
from src.tools import FileTools as ftools
from src.tools import Tools as tools
from src.evaluation import PythonROUGE as PythonROUGE


def load_data(path):
    lines = ftools.read_lines(path)
    data = {}
    for line in lines:
        tmp = line.split(",")
        data[tmp[0]] = [float(tmp[1]),float(tmp[2])]
    return data

def analyze_rate(name = "cleandata_small",num = None):
    save_path = Dir.res+"/result/judge.txt"
    jude_dict = tools.load_object(save_path)
    print(len(jude_dict))

    rate = [0,0,0,0]
    # nums = [int(num*var/sum(rate)) for var in rate]

    textrank_path = Dir.res + "/result/"+name+"/TextRank/detials.txt"
    tr_data = load_data(textrank_path)
    tr_data = dict(sorted(tr_data.items(), key=lambda d: d[1][0], reverse=False))

    entry_path = Dir.res + "/result/"+name+"/EntryBigraph/detials.txt"
    entry_data = load_data(entry_path)
    entry_data = dict(sorted(entry_data.items(), key=lambda d: d[1][0], reverse=False))

    fvae = Dir.res + "/result/"+name+"/Fourth Version auto encoder/detials.txt"
    fvae_data = load_data(fvae)
    fvae_data = dict(sorted(fvae_data.items(), key=lambda d: d[1][0], reverse=True))

    fvsae = Dir.res + "/result/"+name+"/Fourth Version simple auto encoder/detials.txt"
    fvsae_data = load_data(fvsae)
    fvsae_data = dict(sorted(fvsae_data.items(), key=lambda d: d[1][0], reverse=True))

    res = []

    new_dict = dict(sorted(jude_dict.items(),key = lambda d:d[0] ,reverse=True))

    for na in new_dict.keys():
        if num!=None and len(res)>= num:
            break
        if new_dict[na]>0.8:
            res.append(na)

    # for key in list(fvsae_data.keys()):
    #     if jude_dict[key] >0.5:
    #        res.append(key)
    #
    #     if len(res)>=nums[0]:
    #         break
    #
    # for key in fvae_data.keys():
    #     if key not in res:
    #         if jude_dict[key] > 0.5:
    #             res.append(key)
    #     if len(res) > sum(nums[:2]):
    #         break
    #
    # for key in entry_data.keys():
    #     if key not in res:
    #         if jude_dict[key] > 0.5:
    #             res.append(key)
    #     if len(res) > sum(nums[:3]):
    #         break
    #
    # for key in tr_data.keys():
    #     if key not in res:
    #         if jude_dict[key] > 0.5:
    #             res.append(key)
    #     if len(res) == num:
    #         break

    return res

def analyze(main_name,compare_index,name = "cleandata_small"):
    save_path = Dir.res+"/result/judge.txt"
    jude_dict = tools.load_object(save_path)
    # print(list(jude_dict.keys())[0])

    print(len(jude_dict))


    entry_path = Dir.res + "/result/"+name+"/EntryBigraph/detials.txt"
    entry_data = load_data(entry_path)
    first_path = Dir.res + "/result/"+name+"/"+main_name+"/detials.txt"
    first_data = load_data(first_path)
    textrank_path = Dir.res + "/result/"+name+"/TextRank/detials.txt"
    tr_data = load_data(textrank_path)
    result = {}
    for key in first_data.keys():
        a = first_data[key][0] - entry_data[key][0]
        b= first_data[key][1] - entry_data[key][1]
        c = first_data[key][0] - tr_data[key][0]
        d =first_data[key][1] - tr_data[key][1]
        e= first_data[key][0] - tr_data[key][0]+entry_data[key][0]-tr_data[key][0]
        f= first_data[key][1] - tr_data[key][1]+entry_data[key][1]-tr_data[key][1]
        result[key] = [a,b,c,d,e,f]
    count = 0
    news_root = Dir.res+"/"+name+"/news/"
    abst_root = Dir.res+"/"+name+"/abstract/"
    fname  = ftools.get_files(news_root)
    new_result = {}
    for filename in fname:
        # print(filename,count,len(fname))
        # news = ftools.read_lines(news_root+filename)
        # weibo = ftools.read_lines(abst_root+filename)
        # jude = data_filter(news,weibo)
        # jude_dict[filename] = jude
        jude = jude_dict[filename]
        if jude >0.5:
            new_result[filename] = result[filename]
            new_result[filename].append(jude)
            count+=1
    tools.save_object(jude_dict,Dir.res+"/result/judge.txt")
    tmp = dict(sorted(new_result.items(),key = lambda d:d[1][compare_index], reverse=True))
    save_dict = {}
    names = []
    for key in tmp.keys():
        save_dict[key] = tmp[key]
        names.append(key)
    save_path = Dir.res + "/result/"+name+"/"+main_name+".txt"
    ftools.write_com_dict(save_path,save_dict)
    return names

def get_clean_data(filenames,newname,name = "cleandata_small"):
    import shutil
    path = Dir.res + "/result/"+name+"/Fourth Version auto encoder.txt"
    # lines = ftools.read_lines(path)
    # files = []
    # for i in range(len(lines)):
    #     line = lines[i].split(",")
    #     files.append(line[0])
    #     if len(files)>size:
    #         break
    nroot = Dir.res+"/"+name+"/news/"
    aroot = Dir.res+"/"+name+"/abstract/"

     # = "cleandata_highquality_1000"

    if ftools.isexists(Dir.res+"/"+newname+"/"):
        shutil.rmtree(Dir.res+"/"+newname+"/")
    if ftools.isexists(Dir.res + "/result/"+newname+"/"):
        shutil.rmtree(Dir.res+"/result/"+newname+"/")
    snroot = Dir.res+"/"+newname+"/news/"
    saroot = Dir.res+"/"+newname+"/abstract/"

    count = 0
    for name in filenames:
        count+=1
        print(count, len(filenames))
        ftools.copy(nroot+name,snroot+name)
        ftools.copy(aroot+name,saroot+name)

def get_small_data():
    root = Dir.res+"/cleandata_8700/"
    saveroot = Dir.res+"/cleandata_small/"

    flist = ftools.get_files(root+"news/")
    count = 0
    for i in range(len(flist)):
        name = flist[i]
        content = ftools.read_lines(root+"news/"+name)
        if len(content)<80:
            print(count,i,len(flist))
            ftools.copy(root+"news/"+name,saveroot+"news/"+name)
            ftools.copy(root+"abstract/"+name,saveroot+"abstract/"+name)
            count+=1

### news: a list of sentences : list
### abstract: a list of sentences : list
def data_filter(news,abstract):
    if len(news)<10:
        return -2
    if len(abstract)< 2 or len(''.join(abstract))<50:
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

### Second Version
### TextRank
### EntryBigraph
def update_rouge_details(dataname = "cleandata_small",modelname = "EntryBigraph"):
    ref_root = Dir.res+"/"+dataname+"/ref_processed/"
    abs_root = Dir.res+"/result/"+dataname+"/"+modelname+"/abstract_processed/"
    detail_path =Dir.res+"/result/"+dataname+"/"+modelname+"/detials.txt"
    filelist = ftools.get_files(ref_root)
    content = ""
    for i in range(len(filelist)):
        fname = filelist[i]
        print(i,len(filelist))
        abstract = ftools.read_lines(abs_root + fname)
        refence = ftools.read_lines(ref_root + fname)
        lines = [line.split(" ") for line in abstract]
        refen = [line.split(" ") for line in refence]
        rouge1 = rouge_1_simple(refen, lines)
        rouge2 = rouge_2_simple(refen, lines)
        print(fname,rouge1,rouge2)
        content+= fname+","+str(rouge1)+","+str(rouge2)+"\n"

    ftools.write(detail_path,content)

def rouge_1_simple(left, right):
    left_dict, right_dict = {}, {}
    word_count = 0
    for line in left:
        for w in line:
            if w not in left_dict:
                left_dict[w] = 0
            left_dict[w] += 1
            word_count += 1
            if word_count > 140:
                break
    # for key in left_dict.keys():
    #     print(key,left_dict[key])

    word_count = 0
    for line in right:
        for w in line:
            if w not in right_dict:
                right_dict[w] = 0
            right_dict[w] += 1
            word_count += 1
            if word_count > 140:
                break
    # for key in right_dict.keys():
    #     print(key,right_dict[key])

    count = 0
    all = 0
    # print(left_dict.keys())
    for key in left_dict.keys():
        all += left_dict[key]
        if key in right_dict.keys():
            count += min(left_dict[key], right_dict[key])

    return count / all

def rouge_2_simple(left, right):
    left_dict, right_dict = {}, {}
    all = 0
    for line in left:
        for i in range(len(line) - 1):
            w = line[i] + " " + line[i + 1]
            if w not in left_dict:
                left_dict[w] = 0
            left_dict[w] += 1
    for line in right:
        for i in range(len(line) - 1):
            w = line[i] + " " + line[i + 1]
            if w not in right_dict:
                right_dict[w] = 0
            right_dict[w] += 1
    count = 0
    for key in left_dict.keys():
        all += left_dict[key]
        if key in right_dict.keys():
            count += min(left_dict[key], right_dict[key])
    return count / all


def get_result(dataname = "cleandata_highquality_3500"):
    root=  Dir.res+"/result/"+dataname+"/"
    flist = ftools.get_files(root)
    content = ""
    for name in flist:
        if ".txt" in name:
            continue
        lines = ftools.read_lines(root+name+"/eval_res.txt")
        content += name+", "+lines[1][lines[1].index("[")+1:lines[1].index("]")]+"\n"
    print(content)
    ftools.write(Dir.res+"/result/"+dataname+"/result.txt",content)

def rouge_all(filenames):
    fn_root=Dir.res+"/result/cleandata_small/FirstNSentencesSummarizor/abstract_processed/"
    tr_root=Dir.res+"/result/cleandata_small/TextRank/abstract_processed/"
    ii_root=Dir.res+"/result/cleandata_small/EntryBigraph/abstract_processed/"
    ae_root =Dir.res+"/result/cleandata_small/Fourth Version auto encoder/abstract_processed/"
    se_root = Dir.res+"/result/cleandata_small/Fourth Version simple auto encoder/abstract_processed/"

    ref_root =Dir.res+"/cleandata_small/ref_processed/"

    ref_list = []
    fn_list, tr_list, ii_list, ae_list, se_list = [], [], [], [],[]
    for i in range(len(filenames)):
        ref_list.append([ref_root+filenames[i]])
        fn_list.append(fn_root+filenames[i])
        tr_list.append(tr_root+filenames[i])
        ii_list.append(ii_root+filenames[i])
        ae_list.append(ae_root+filenames[i])
        se_list.append(se_root+filenames[i])
    order_n = 4
    print("fn")
    fn_res = PythonROUGE.PythonROUGE(fn_list,ref_list,ngram_order = order_n)
    print(fn_res)
    print("tr")
    tr_res = PythonROUGE.PythonROUGE(tr_list,ref_list,ngram_order = order_n)
    print(tr_res)
    print("ii")
    ii_res = PythonROUGE.PythonROUGE(ii_list,ref_list,ngram_order = order_n)
    print(ii_res)
    print("ae")
    ae_res = PythonROUGE.PythonROUGE(ae_list,ref_list,ngram_order = order_n)
    print(ae_res)
    print("se")
    se_res = PythonROUGE.PythonROUGE(se_list,ref_list,ngram_order = order_n)
    print(se_res)


if __name__ == "__main__":

    # update_rouge_details(dataname="cleandata_small",modelname="TextRank")
    # update_rouge_details(dataname="cleandata_small",modelname="Second Version")
    # update_rouge_details(dataname="cleandata_small",modelname="EntryBigraph")

    # get_small_data()

    res = analyze_rate()
    print(len(res))
    rouge_all(res)
    # get_clean_data(res, newname="cleandata_highquality_" + str(len(res)))

    # main_name = "Fourth Version auto encoder"
    # for i in range(4,6):
    #     compare_index = i
    #
    #     filenames = analyze(main_name,compare_index)
    #     num = 2000
    #
    #     print(main_name,compare_index,num)
    #     rouge_all(filenames[:num])
    #     print("-----------------------")
    #     main_name = "Fourth Version simple auto encoder"
    #     print(main_name, compare_index,num)
    #     filenames = analyze(main_name,compare_index)
    #     rouge_all(filenames[:num])
    # size = 2000
    # get_clean_data(filenames[:num],newname="cleandata_highquality_"+str(num))

    # get_result("cleandata_highquality_1500")
    pass
