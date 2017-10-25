from src.tools import Tools as tools
from src.models.Summarizor import Summarizor
from src.vectorizer.auto_vector import Auto_Simple_Vec as ASVec
from src.vectorizer.two_layer_autoencoder_vector import AutoCoder as AEVec
import Dir


class FastSummarize():

    def __init__(self,Vec):
        self.summ = Summarizor(Vec)
        self.name = "Fast "+self.summ.name
        self.path = Dir.res+"/encoder/"+self.summ.name+"/"
        self.parameter ={}
        self.seperate = "#$#"
        self.info = self.name + ", weight=" + str(
            self.summ.weight) + ", vector="+ str(self.summ.vectorizer.name)+" , distance="+str(self.summ.disttype)
        self.load_parameter()

    def set_weight(self,weight):
        self.summ.weight = weight
        # self.name = "FFV"+'_'.join(map(str,self.summ.weight))
        self.info = self.name + ", weight=" + str(
            self.summ.weight) + ", vector=" + str(self.summ.vectorizer.name) + " , distance=" + str(self.summ.disttype)

    def load_parameter(self):
        flist = tools.get_files(self.path)
        # print(flist)
        for name in flist:
            fpath = self.path+name
            tmp = tools.load_object(fpath)
            for key in tmp.keys():
                self.parameter[key] = tmp[key]

    def summarize(self,text,num =3,fname=None):

        # print(len(self.parameter.keys()))
        # for key in self.parameter.keys():
        #     print(key)
        # print(text)
        if fname in self.parameter.keys():
            sens, sens_words, sens_tag = self.summ.analyze(text)
            coverage_list = self.parameter[fname][0]
            relative_matrix= self.parameter[fname][1]
            clues_list    = self.parameter[fname][2]
            entities_list = self.parameter[fname][3]
            options = self.summ.generate_options(len(sens), num)
            max_value =0
            best_option =[]
            for option in options:
                option_score,tmp = self.summ.score_option(option, coverage_list, relative_matrix, clues_list, entities_list)

                # tmp.append(str(option_score))
                if option_score > max_value:
                    best_option = option
                    max_value = option_score
            abstract = [sens[var] for var in best_option]
            # print('\n'.join(tmp),max_value)
            return abstract
        else:
            print("using original summarizor")
            return self.summ.summarize(text,num)


if __name__ == "__main__":
    from src.tools import  FileTools as ftools
    test_file = Dir.res + "/cleandata_highquality_100/news/trainning_31.txt"
    text = ftools.read_lines(test_file)
    summ = FastSummarize(ASVec)
    print(summ.info)
    res = summ.summarize(text)
    for line in res:
        print(line)

    test_file = Dir.res + "/cleandata_highquality_100/news/trainning_32.txt"
    text = ftools.read_lines(test_file)
    summ = FastSummarize(ASVec)
    print(summ.info)
    res = summ.summarize(text)
    for line in res:
        print(line)