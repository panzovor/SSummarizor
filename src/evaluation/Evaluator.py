import time
from src.tools import FileTools as tools
import src.evaluation.PythonROUGE
import Dir
import src.ResultProcess.ResultPropress as RP
import multiprocessing
from src.baseline.TextRank import  TextRank
from src.baseline.FirstNSenetencesSummarizor import FirstNSentencesSummarizor as FirstNSentencesSummarizor
from src.baseline.EntryBiGraph import EntryBiGraph
from src.models.FourthVersion import  Summarizor as Summarizor4
# from src.models.ThirdVersion import  Summarizor as Summarizor3
from src.models.SecondVersion import  Summarizor as Summarizor2
from src.models.Summarizor import Summarizor as Summarizor
from src.models.FastFirstVersion import FastSummarize as FSummarizor
from src.vectorizer.auto_vector import Auto_Simple_Vec as ASVec
from src.vectorizer.two_layer_autoencoder_vector import AutoCoder as AEVec
from src.models.uperbond import  Uper as Uper
from src.evaluation import PythonROUGE as PythonROUGE

import warnings
warnings.filterwarnings("ignore")

def workers(args):
    wname = args["n"]
    data = args["d"]
    model = args["m"]
    abstract = args["a"]
    count = 0
    summarize_result = {}
    print(wname, "start")
    for text in data.keys():
        # if text not in summarize_result.keys():
        start = time.time()
        # try:
        summarize_result[text] = model.summarize(data[text], num=3,fname = text)
        count += 1
        end = time.time()
        tools.write_list(abstract + text + ".txt", summarize_result[text])
        # print(wname, text, count, "/", len(data), end - start)
        # except:
        #     print(wname,text)
        #     input()
    # print(wname,len(self.summarize_result))
    print(wname, " done")
    return summarize_result

class Evaluator:


    def __init__(self,corpus,corpus_ref,reprocess = True,parall = False,cpu = 4):
        self.file = corpus
        self.file_ref = corpus_ref
        self.data = tools.read_dir_lines_dict(self.file)
        self.rouge=  src.evaluation.PythonROUGE
        self.word_index={}
        self.dir_path = corpus[:corpus.rindex("/")]
        self.ref_processed = self.dir_path + "/ref_processed/"
        self.ref_seperate = self.dir_path + "/ref_seperate/"
        self.indexlize_data(reprocess=reprocess)
        self.start =0
        self.parall = parall
        self.cpu = 4

    def indexlize_data(self,reprocess):
        ###  建立词到数值的映射
        print("start")
        word_index_path = self.dir_path + "/words_index.txt"
        if not tools.isexists(word_index_path) or \
                not tools.isexists(self.ref_processed) or \
                not tools.isexists(self.ref_seperate) or \
                len(tools.get_files(self.ref_seperate)) ==0 or \
                len(tools.get_files(self.ref_processed)) == 0:
            reprocess = True
        if reprocess:

            self.word_index = RP.build_word_index(self.file, word_index_path)
            # print("word_index_builded")
            ###  参考摘要数值化
            print(self.file_ref,self.ref_seperate,self.ref_processed)
            RP.result_process(self.file_ref, self.ref_seperate)
            RP.replace_words_by_num(self.word_index, self.ref_seperate, self.ref_processed)
            print("references process done")
        else:
            self.load_word_index(word_index_path )
            print("word index loaded")

    def load_word_index(self,path):
        lines = tools.read_lines(path)
        for line in lines:
            index = line.rindex(":")
            # print(line[:index])
            # print(line[index+1:])
            self.word_index[line[:index]] = int(line[index+1:])
        # print(self.word_index["转嫁"])



    def evaluator_rouge(self,model,result_dir,num):
        summarize_result={}
        astart = time.time()
        ### 保存模型的摘要结果
        abstract = result_dir+"/abstract/"
        keys = sorted(self.data.keys())
        if self.parall:
            p = multiprocessing.Pool(self.cpu)
            inter = int(len(keys) / self.cpu) + 1
            args = []
            for i in range(self.cpu):
                tmp = {}
                if i == 0:
                    key = keys[:inter]
                    # print(i,"0",inter,len(key))
                    for k in key:
                        tmp[k] = self.data[k]
                elif i == self.cpu-1:

                    key = keys[i*inter:]
                    # print(i, i * inter, "end", len(key))
                    for k in key:
                        tmp[k] = self.data[k]
                else:

                    key = keys[i*inter:(i+1)*inter]
                    # print(i, i * inter, (i + 1) * inter, len(key))
                    for k in key:
                        tmp[k] = self.data[k]
                args.append({"n":"work"+str(i),
                            "d":tmp,
                            "m":model,
                             "a" :abstract
                             }
                            )
            # input()
            rslt = p.map(workers,args)
            # for var in rslt:
            #     for k in var.keys():
            #         summarize_result[k] = var[k]
            ### 处理摘要结果数据（数值化）
            # print("saving abstract ",len(summarize_result))
            # for fname in summarize_result.keys():
                # tools.write_list(abstract + fname + ".txt", summarize_result[fname])
            abstract_processed = result_dir + "/abstract_processed/"
            abstract_seperate = result_dir + "/abstract_seperate/"
            RP.result_process(abstract, abstract_seperate)
            print("abstract separate done")
            RP.replace_words_by_num(self.word_index, abstract_seperate, abstract_processed)
            print("abstract replace done")
            # print(abstract_processed,result_dir)
            self.rouge_detail(abstract_processed, result_dir)

            ### 计算 ROUGE
            # import src.evaluation.ROUGE
            # self.rouge = src.evaluation.ROUGE.ROUGE()
            # print("evaling")
            result = self.rouge.eval(abstract_processed, self.ref_processed,num)
            eval_result = result_dir + "/eval_res.txt"
            print(result)
            tools.write(eval_result, model.info + "\n" + result, mode="a")
            aend = time.time()
            print(aend - astart)

        else:
            count = 0
            for text in keys:
                if text not in summarize_result.keys():
                    start = time.time()
                    count+=1
                    # if count <1530:
                    #     count+=1
                    #     continue
                    # print(text)
                    summarize_result[text] = model.summarize(self.data[text], num,fname = text)


                    end = time.time()
                    # tools.print_proccess(count, len(self.data.keys()))
                    print(text,count,"/",len(keys),end-start)
                    # print(result_save_dir_abstract + text + ".txt")
                    # print(  model.summarize(self.data[text], num) )
                    tools.write_list(abstract + text + ".txt", summarize_result[text])
            ### 处理摘要结果数据（数值化）
            abstract_processed = result_dir+"/abstract_processed/"
            abstract_seperate = result_dir + "/abstract_seperate/"
            RP.result_process(abstract,abstract_seperate)
            RP.replace_words_by_num(self.word_index,abstract_seperate,abstract_processed)
            # print(abstract_processed,result_dir)
            self.rouge_detail(abstract_processed,result_dir)

            ### 计算 ROUGE
            # import src.evaluation.ROUGE
            # self.rouge = src.evaluation.ROUGE.ROUGE()
            # print("evaling")
            result = self.rouge.eval(abstract_processed, self.ref_processed)
            eval_result = result_dir+"/eval_res.txt"
            print(result)
            tools.write(eval_result,model.info+"\n"+result,mode="a")
            aend = time.time()
            print(aend-astart)

    ### left is reference
    ### right is abstract generated by model
    def rouge_1_simple(self, left, right):
        left_dict, right_dict = {}, {}
        word_count = 0
        for line in left:
            for w in line:
                if w not in left_dict:
                    left_dict[w] = 0
                left_dict[w] += 1
                word_count+=1
                if word_count>140:
                    break
        word_count = 0
        for line in right:
            for w in line:
                if w not in right_dict:
                    right_dict[w] = 0
                right_dict[w] += 1
                word_count += 1
                if word_count>140:
                    break
        count = 0
        all = 0
        # print(left_dict.keys())
        for key in left_dict.keys():
            all+= left_dict[key]
            if key in right_dict.keys():
                count += min(left_dict[key], right_dict[key])

        return count/all

    ### left is reference
    ### right is abstract generated by model
    def rouge_2_simple(self, left, right):
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
        return count/all

    def rouge_detail(self,abstract_processed,save_dir):
        flist = tools.get_files(abstract_processed)
        save_content = []
        for fname in flist:
            content = tools.read_lines(abstract_processed+fname)
            refence = tools.read_lines(self.ref_processed+fname)
            lines =[line.split(" ") for line in content]
            refen =[line.split(" ") for line in refence]
            rouge1 = self.rouge_1_simple(refen,lines)
            rouge2 = self.rouge_2_simple(refen, lines)
            save_content.append(fname+","+str(rouge1)+","+str(rouge2))
        tools.write_list(save_dir+"/detials.txt",save_content)



    def test(self,model,saveroot):
        print(model.info)
        result_save = saveroot + model.name
        self.evaluator_rouge(model, result_save, num=4)

class Evaluator_part():

    def __init__(self):
        pass

    def eval(self,ref_root,guess_root,flist):
        guess_list, ref_list = [],[]
        for f in flist:
            ref_path = ref_root+f
            gue_path = guess_root+f
            ref_list.append([ref_path])
            guess_list.append(gue_path)

        print(len(guess_list),len(ref_list))
        recall_list, precision_list, F_measure_list = PythonROUGE.PythonROUGE(guess_list,ref_list)
        string = ""
        string += 'recall = ' + str(recall_list) + '\n'
        string += 'precision = ' + str(precision_list) + '\n'
        string += 'F = ' + str(F_measure_list) + '\n'

        # print(string)
        return string


    def test_small_data(self,modelname,flist):
        ref_root= Dir.res+"/cleandata_small/ref_processed/"
        guess_root=  Dir.res+"/result/cleandata_small/"+modelname+"/abstract_processed/"
        res = self.eval(ref_root,guess_root,flist)
        print(res)

if __name__ == "__main__":
    ### initial test data
    name = "cleandata_highquality_1640"
    corpus = Dir.res + "/"+name+"/news"
    corpus_ref = Dir.res + "/"+name+"/abstract"
    saveroot=Dir.res + "/result/"+name+"/"
    eva = Evaluator(corpus,corpus_ref,False,parall=True)
    print(name)

    ### First N Sentence
    # eva.test(FirstNSentencesSummarizor(), saveroot=saveroot)

    ### get the upper limit of abstract
    # eva.test(Uper(),saveroot=saveroot)

    ### parametering textrank
    # from src.tools.Tools import  Distance as Distance
    # tr= TextRank()
    # alpha = [0.2,0.4,0.6,0.8]
    # for type_ in [Distance.COS,Distance.EUD,Distance.OCCLOSE]:
    #     for al in alpha:
    #         tr.set_simi(type_,None,alpha=al)
    #         print(tr.info)
    #         eva.test(tr,saveroot=saveroot)
    # #
    # # tr.set_simi(Distance.COS,None,alpha=0.8)
    # print(tr.info)
    # eva.test(tr,saveroot=saveroot)
    #
    # tr.set_simi(Distance.OCCLOSE,None,alpha=0.8)
    # print(tr.info)
    # eva.test(tr,saveroot=saveroot)




    ## parametering entityBigraph
    # en= EntryBiGraph()
    # targets = [["n"],["n","v"],["n","v","m"],["all"],["all_n"],["n","m"]]
    # targets = [["all_n","m"],["all_n","v"],["all_n","m","v"]]
    # # for tt in targets:
    # for i in range(len(targets)):
    #     tt = targets[i]
    #     en.set(targets=tt)
    #     eva.test(en,saveroot=saveroot)

    ### parametering auto encoder
    import gc
    from src.tools.Tools import Distance as Distance
    # Encoder = ASVec
    Encoder = AEVec
    summ = Summarizor(Encoder)
    itimes = [20,40,60,80]
    # dista = [Distance.COS,Distance.EUD]
    dista = [Distance.EUD]
    for dis in dista:
        for i in range(len(itimes)):
            gc.collect()
            print("gc collect done")
            summ.set_parameter(itimes[i],dis)
            eva.test(summ, saveroot=saveroot)
    #
    # summ = FSummarizor(Encoder)
    # for i in range(4):
    #     weight = [1,1,1,1]
    #     gc.collect()
    #     weight[i] = 0
    #
    #     summ.set_weight(weight)
    #     eva.test(summ, saveroot=saveroot)