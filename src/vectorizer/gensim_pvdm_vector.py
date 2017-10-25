from gensim.models.doc2vec import Doc2Vec as D2V
from gensim.models.doc2vec import TaggedDocument as TaggedDocument
import Dir
from src.tools import FileTools as ftools
from src.tools import Tools as tools
import numpy as np
import pickle

class Sen2Vec():

    def __init__(self,size =200, window = 5,min_count =1):
        self.model = None
        self.size = size
        self.window = window
        self.min_count = min_count
        self.save_path =  Dir.res+"/pvdm/sen_model.model"
        self.vect_path = Dir.res+"/pvdm/sen_vector.pkl"
        self.sen_dict= {}

    def train(self,path = Dir.res + "/cleandata_604/news/"):
        data = self.load_data(path)
        print("data loaded")
        self.model = D2V(data,size= self.size,window=self.window,min_count=self.min_count)

        for key in self.sen_dict.keys():
            self.sen_dict[key] = np.array(self.model.docvecs[self.sen_dict[key]])
        self.save()

    def load(self,path = None,vec_only= False):
        if path == None:
            path = self.save_path
        if not vec_only:
            self.model = D2V.load(path)
        with open(self.vect_path, 'rb') as handle:
            self.sen_dict = pickle.load(handle)
            # print(list(self.sen_dict.keys())[0])
            # print(self.sen_dict[list(self.sen_dict.keys())[0]])

    def save(self):
        self.model.save(self.save_path)
        # print(list(self.sen_dict.keys())[0])
        # print(self.sen_dict[list(self.sen_dict.keys())[0]])
        with open(self.vect_path, 'wb') as handle:
            pickle.dump(self.sen_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)



    def load_data(self,path = Dir.res + "/cleandata_604/news/"):
        flist = ftools.get_files(path)
        data = []
        count =0
        for name in flist:
            filepath = path + name
            lines = ftools.read_lines(filepath)
            for line in lines:
                words = tools.seperate(line)
                data.append(TaggedDocument(words, ["sen_"+str(count)]))
                self.sen_dict[''.join(words)] =  "sen_"+str(count)
                count+=1
        return data

    def get_sen_vec(self,sens):
        if ''.join(sens) in self.sen_dict.keys():
            return self.sen_dict[''.join(sens)]
        else:
            return self.model.infer_vector(sens)


class Doc2Vec(Sen2Vec):
    def __init__(self,size =200, window = 5,min_count =1):
        self.model = None
        self.size = size
        self.window = window
        self.min_count = min_count
        self.save_path = Dir.res + "/pvdm/d2v_model.model"
        self.vect_path = Dir.res + "/pvdm/d2v_vector.pkl"
        self.sen_dict = {}

    def load_data(self,path = Dir.res + "/cleandata_604/news/"):
        flist = ftools.get_files(path)
        data = []
        count =0
        for name in flist:
            filepath = path + name
            lines = ftools.read_lines(filepath)
            essay = ""
            tmp = []
            for line in lines:
                words = tools.seperate(line)
                tmp.extend(words)
                essay+=''.join(words)
            data.append(TaggedDocument(tmp, ["text_"+str(count)]))
            self.sen_dict[essay] =  "text_"+str(count)
            count+=1
        return data

class pvdm_vectorize():

    def __init__(self):
        self.name = "pvdm vectorizer"
        self.sen2v = Sen2Vec()
        self.doc2v = Doc2Vec()
        self.sen2v.load()
        self.doc2v.load()

    def vectorize(self,sens,sens_tag = None):
        sens_vect=[]
        essay_key = []
        for sen in sens:
            essay_key .extend(sen)
            vec = self.sen2v.get_sen_vec(sen)
            # if vec == None:
            #     input()
            sens_vect.append(vec)
        essay_vector = self.doc2v.get_sen_vec(essay_key)
        return sens_vect,essay_vector

if __name__ == "__main__":

    # sen2v = Sen2Vec()
    # sen2v.train()
    # doc2v= Doc2Vec()
    # doc2v.train()

    sens = ftools.read_lines(Dir.res+"/cleandata_604/news/training_4.txt")
    pvdm_v = pvdm_vectorize()
    text =[]
    for line in sens:
        text.append(tools.seperate(line))
    sens,essay = pvdm_v.vectorize(text)
    print(sens[0])
    for ss in sens:
        print(ss)

    # print(essay)
