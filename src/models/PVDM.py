import gensim
import numpy as np
from gensim.models.doc2vec import Doc2Vec,LabeledSentence
import Dir
import src.tools.FileTools as tools
import re
import random
import math
LabeledSentence = gensim.models.doc2vec.LabeledSentence

class PVDM_Summariztion():

    def __init__(self):
        self.name = "pvdm_summariztion"
        self.dir = Dir.resource+"/extract_data_process/data1000/news/"
        self.vector_dict = None

    def label_data(self,data):
        labelized =[]
        for filename in data.keys():
            for i in range(data[filename].__len__()):
                sentence = data[filename][i]
                labelized.append(LabeledSentence(sentence, [filename+"_"+str(i)]))
        return labelized

    def train_sentence_vector(self,x_train,size=400, epoch_num=10):
        # 实例DM和DBOW模型
        model_dm = gensim.models.Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, workers=3)
        model_dbow = gensim.models.Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, dm=0, workers=3)

        model_dm.build_vocab(x_train)
        model_dbow.build_vocab(x_train)

        # 进行多次重复训练，每一次都需要对训练数据重新打乱，以提高精度
        for epoch in range(epoch_num):
            random.shuffle(x_train)

            model_dm.train(x_train)
            model_dbow.train(x_train)
        return model_dm,model_dbow

    def seperate_sentences(self, essay):
        regex = "。|？|！|；|\n"
        # print(essay)
        sentences = re.split(regex, essay)
        result = []
        for sen in sentences:
                # print("ddd",sen.strip())
            result.append(sen.strip())
        return result

    def getVecs(self,model, corpus, size):
        vecs = [np.array(model.docvecs[z.tags[0]]).reshape((1, size)) for z in corpus]
        return np.concatenate(vecs)

    def transfer(self,x_traindata,size = 400):
        dm, dbow = self.train_sentence_vector(x_traindata,size=size)
        train_vecs_dm = self.getVecs(dm, x_traindata, size)
        train_vecs_dbow = self.getVecs(dbow, x_traindata, size)
        train_vecs = np.hstack((train_vecs_dm, train_vecs_dbow))
        result = {}
        for i in range(x_traindata.__len__()):
            result[x_traindata[i][0]] = train_vecs[i]
        return result

    def read_file(self,dir):
        filelist = []
        tools.get_filelist(dir,filelist)
        data = {}
        reverse_data={}
        filelist = sorted(filelist)
        for filename in filelist:
            with open(filename,mode= "r",encoding= "utf-8") as file:
                content = file.read()
                sentences = self.seperate_sentences(content)
                data[filename] = sentences
                for sen in sentences:
                    if sen not in reverse_data.keys():
                        reverse_data[sen]= [tools.get_name(filename)]
                    else:

                        reverse_data[sen].append(tools.get_name(filename))
                        # print(sen,reverse_data[sen])
        return data,reverse_data

    def similarity(self,v1,v2):
        if v1.__len__() != v2.__len__():
            return None
        fenzi = [v1[i]*v2[i] for i in range(v1.__len__())]
        fenmu_v1 = [v1[i]*v1[i] for i in range(v1.__len__())]
        fenmu_v2 = [v2[i]*v2[i] for i in range(v2.__len__())]
        return sum(fenzi) / (math.sqrt(sum(fenmu_v1))* math.sqrt(sum(fenmu_v2)))

    def get_similarity(self,sentences,vector_dict):
        simi = []
        for i in range(sentences.__len__()):
            tmp = []
            for j in range(sentences.__len__()):
                tmp.append(self.similarity(vector_dict[sentences[i]],vector_dict[sentences[j]]))
            simi.append(tmp)
        return simi

    def cluster(self,sentences,vector_dict,num = 3,learning_rate =0.01,thredshold = 0.2):
        Cluster =[]
        repeat = max([int(thredshold/learning_rate),int((1-thredshold)/learning_rate)])
        time =0
        simi_matrix = self.get_similarity(sentences,vector_dict)
        # for res in simi_matrix:
        #     print(res)
        while Cluster.__len__()!= num:
            if time > repeat:
                break
            Cluster = []
            for i in range(sentences.__len__()-1):
                if Cluster.__len__() == 0:
                    Cluster.append([i])
                else:
                    cluster_simi =[]
                    for j in range(Cluster.__len__()):
                        cluster_ = Cluster[j]
                        simi_all = sum([self.similarity(vector_dict[sentences[i]],vector_dict[sentences[index]]) for index in cluster_])/cluster_.__len__()
                        # simi_all = sum([simi_matrix[i][index] for index in cluster_])/cluster_.__len__()
                        cluster_simi.append(simi_all)
                    max_cluster = cluster_simi.index(max(cluster_simi))
                    if max(cluster_simi)> thredshold:
                        Cluster[max_cluster].append(i)
                    else:
                        Cluster.append([i])
            time+=1
            if Cluster.__len__()>num:
                thredshold-=learning_rate
            else:
                thredshold+=learning_rate
        return Cluster

    def option_generator(self,Cluster,sentences,vector_dict,num = 2):
        options  = []
        tmp_selection = []
        middles = []
        for cluster_ in Cluster:
            vector = []
            for i in cluster_:
                vector.append(np.array(vector_dict[sentences[i]]))
            middle = sum(vector)/vector.__len__()
            middles.append(middle)
            if cluster_.__len__() <= num:
                tmp_selection.append(cluster_)
                continue

            simi_value= []
            for i in cluster_:
                v = vector_dict[sentences[i]]
                simi_value.append(self.similarity(v,middle))
            selected_index = []
            for i in range(num):
                max_index= simi_value.index(max(simi_value))
                selected_index.append(max_index)
                simi_value[max_index]=  -1
            tmp_selection.append(selected_index)
        self.select_options(tmp_selection, options)
        return options,middles

    def optimization (self,options,sentences,vector_dict,middle,r = 0.5):
        result = []
        coverage = []
        relative = []
        for option in options :
            tmp =[]
            tmp1 = []

            for i in range(option.__len__()):
                tmp.append(self.similarity(middle[i],vector_dict[sentences[option[i]]]))
                for j in range(i,option.__len__()):
                    if i!= j :
                        tmp1.append(self.similarity(vector_dict[sentences[option[i]]],vector_dict[sentences[option[j]]]))
            try:
                coverage.append(sum(tmp)/tmp.__len__())
                relative.append(sum(tmp1)/tmp1.__len__())
            except:
                print(option)
                print(options)
                print(tmp)
                print(tmp1)

                input()

        for i in range(options.__len__()):
            result.append((1-r)*coverage[i]+r*relative[i])
        return options[result.index(max(result))]


    def select_options(self,data,result, i=0, tmp=None):
        i += 1
        if i - 1 == data.__len__():
            result.append(tmp)
            return True
        for ele in data[i - 1]:
            if tmp == None:
                tmp = []
            if tmp.__len__() == i:
                tmp1 = tmp[:i - 1]
                tmp = tmp1
            elif tmp.__len__() > i:
                tmp1 = tmp[:i - 1]
                tmp = tmp1
            tmp.append(ele)
            self.select_options(data,result,  i, tmp)


    def train(self,size):
        print("train sentence vector")
        data,reverse_data = self.read_file(self.dir)
        labeled_sentences = self.label_data(data)
        result = self.transfer(labeled_sentences, size=size)
        self.vector_dict = result
        print("trian done")
        save_path = self.save_vector(self.vector_dict,size)
        self.vector_dict = self.load_vector(save_path)
        return result

    def save_vector(self,vectors,name = None,save_path = Dir.resource+"/pv_dm_vector/vectors"):
        content = ""
        if name !=None:
            save_path = save_path+str(name)
        for sentence in vectors.keys():
            content+= sentence+"<###>"+str(list(vectors[sentence]))[1:-1]+"\n"
        with open(save_path,mode="w",encoding="utf-8") as file:
            file.write(content)
        return save_path

    def load_vector(self,load_path = Dir.resource+"/pv_dm_vector/vectors"):
        result ={}
        with open(load_path,mode="r",encoding="utf-8") as file:
            for line in file.readlines():
                tmp = line.split("<###>")
                if tmp[0] not in result.keys():
                    result[tmp[0]] = np.array([float(value) for value in tmp[1].split(",")])
        return result

    def summarize(self,content,num =3,size= 50,r_factor = 0.1):
        if self.vector_dict == None:
            # self.vector_dict = self.load_vector()
            self.train(size)
        if not isinstance(content,list):
            sentences = self.seperate_sentences(content)
        else:
            sentences = content

        cluster_result = self.cluster(sentences,vector_dict= self.vector_dict,num=num)
        options,middle = self.option_generator(cluster_result,sentences,self.vector_dict)
        optimization_option = self.optimization(options,sentences,self.vector_dict,middle,r=r_factor)
        return [sentences[i] for i in optimization_option]

    def demo(self):
        essay = Dir.resource+"/extract_data_process/data1000/single_text/training_246.txt"
        with open(essay,mode="r",encoding="utf-8") as file:
            content = file.read()
            result= self.summarize(content)
        for line in result:
            print(line)

# demo()
if __name__ =="__main__":
    pvdm = PVDM_Summariztion()
    pvdm.demo()

