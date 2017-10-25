import time
from src.tools.Tools import  Dist as Dist
from src.tools.Tools import  Distance as Distance
import Dir
from src.tools import FileTools as ftools
from src.tools import Tools as tools
# from src.vectorizer.graph_vector import Graph_Vec  as MyVec
from src.vectorizer.auto_vector import Auto_Simple_Vec as MyVec

class Summarizor():

    def __init__(self):
        self.name = "Fourth Version"
        self.vectorizer = MyVec()
        self.dist = Dist()
        self.disttype =Distance.EUD
        self.weight = [1,1,1,1]
        self.count_index = 0
        self.info = self.name+", weight="+str(self.weight)+", vector="+self.vectorizer.name+", distance="+str(self.disttype)
        self.cluewords =set()
        self.load_clue_words()
        self.target_tag = ["n","v","m"]
        self.clue_weight = 0

    def set_weight(self,weight):
        self.weight = weight
        self.info = self.name + ", weight=" + str(
            self.weight) + ", vector=" + self.vectorizer.name + ", distance=" + str(self.disttype)
        self.name = "SV"+'_'.join(map(str,weight))

    def load_clue_words(self,path= Dir.res+"/parameter/summarization_parameter/clue_words"):
        list1 = ftools.read_lines(path)
        for var in list1:
            self.cluewords.add(var.strip())

    '''
    calculate the coverage value list
    input : sens_vector:   a list vector of sentences 
            essay_vector:  the vector of essay
    output: coverage_list: a coverage value list of each sentence vector
    '''
    def coverage_values(self,sens_vector,essay_vector):
        coverage_list =[]
        for sv in sens_vector:
            coverage_list.append(self.dist.sim(sv,essay_vector,self.disttype))

        tools.normarlize(coverage_list)

        return coverage_list

    '''
    calculate the relative value matrix
    input : sens_vector:     a vector list of sentences
    output: relative_matrix: a relative value matrix of each two sentences
    '''
    def relative_values(self,sens_vector):
        relative_matrix=[[0]*len(sens_vector) for var in range(len(sens_vector))]
        for i in range(len(sens_vector)-1):
            for j in range(i+1,len(sens_vector)):
                relative_matrix[i][j] = max(self.dist.sim(sens_vector[i],sens_vector[j],self.disttype),0)
                relative_matrix[j][i] = relative_matrix[i][j]
        return relative_matrix

    '''
    calcluate the clue value of each sentences
    input : sens_words: a words list of each sentence
    output: clues_list: a clues value list of each sentences 
    '''
    def clueswords_values(self,sens_words):
        clue_list=  [0]*len(sens_words)
        words =[]
        for var in sens_words:
            for w in var:
                words.append(w)
        sen_len_list= []
        for i in range(len(sens_words)):
            sen_w = sens_words[i]
            sen_len_list.append(len(sen_w)/len(words))
            for w in sen_w:
                if w in self.cluewords:
                    clue_list[i]=1
                    break
        tools.normarlize(sen_len_list)
        for i in range(len(sens_words)):
            clue_list[i] =self.clue_weight*clue_list[i] + (1-self.clue_weight)*sen_len_list[i]
        return clue_list

    '''
    calcluate the entities list of each sentences(not duplicated)
    input : sens_words:    a words list of each sentences
            sen_tag   :    a words'tag list of each sentences
    output: entities_list: a entities words list of each sentences
    '''
    def entities_values(self,sens_words,sen_tag):
        entities_list =[]
        for i in range(len(sens_words)):
            tmp = []
            for j in range(len(sens_words[i])):
                if sen_tag[i][j][0] in self.target_tag[0]:
                    tmp.append(sens_words[i][j])
            entities_list.append([tmp,sens_words[i]])
        return entities_list

    '''
    calculate the score of input option
    input: option         : the setences index list of option
           coverage_list  : the coverage value list of each sentences
           relative_matrix: the relative value betweent each two sentences
           clues_list     : the clues value list of each sentences
           entities_list  : the entities words list of each sentences
    output: score         : the score of input option
    '''
    def score_option(self,option,coverage_list,relative_matrix,clues_list,entities_list):
        coverage_value,relative_value,clues_value,entities_value =0,0,0,0
        tmp,word = set(),set()
        for i in range(len(option)):
            coverage_value+= coverage_list[option[i]]
            for j in range(len(option)):
                if i!= j:
                    relative_value += relative_matrix[option[i]][option[j]]
            clues_value += clues_list[option[i]]
            for var in entities_list[option[i]][0]:
                tmp.add(var)
            for var in entities_list[option[i]][1]:
                word.add(var)
            # print(entities_list[option[i]][0])
            # print(entities_list[option[i]][1])
            # input()
        #
        # print(len(tmp),tmp)
        # print(len(word),word)
        # input()
        coverage_value/= len(option)
        relative_value = 1- relative_value/(len(option)*(len(option)-1))
        entities_value = len(tmp)/len(word)
        clues_value /= len(option)
        # print(coverage_value,relative_value,clues_value,entities_value)
        score = self.weight[0] * coverage_value + self.weight[1] * relative_value \
                + self.weight[2] * clues_value + self.weight[3] * entities_value
        return score,[coverage_value,relative_value,clues_value,entities_value]

    '''
    generate all possible options
    input : len: the length of sentences
            num: the length of abstract
    output: all possible options of abstract
    '''
    def generate_options(self, len, num,start =0):
        res = []
        if num == 1:
            for var in range(start, len):
                res.append([var])
            return res
        for i in range(start, len - num + 1):
            for var in self.generate_options( len, num - 1,i + 1):
                res.append([i] + var)
        return res

    '''
    seperate text into sentences and seperate and tag each sentences
    input : text: the original text
    output: sens      : the sentences of text
            sens_words: the words of each sentences
            sens_tag  : the words's tag of each setences
    '''
    def analyze(self,text):
        sens_words,sens_tag = [],[]
        sens = tools.seperate_sentences(text)
        tmp =[]
        for sen in sens:
            if "原标题" in sen :
                continue
            tmp.append(sen)
            tmp_words,tmp_tag = tools.seperate_pog(sen)
            sens_words.append(tmp_words)
            sens_tag.append(tmp_tag)
        return tmp,sens_words,sens_tag

    def save_value(self,path,text,coverage_list,relative_matrix,clues_list,entities_list):
        ftools.check_filename(path)
        save_dict = {}
        save_dict['#$#'.join(text)] =[coverage_list,relative_matrix,clues_list,entities_list]
        tools.save_object(save_dict,path)

    def summarize(self,text,num =3):
        # print(self.info)
        sens,sens_words,sens_tag = self.analyze(text)
        start = time.time()
        sen_vector,essay_vector = self.vectorizer.vectorize(sens_words,sens_tag)
        end = time.time()
        # print(end-start,"vector")
        # for eee in sen_vector:
        #     print(eee)
        #
        # print(essay_vector)
        self.count_index+=1
        coverage_list = self.coverage_values(sen_vector,essay_vector)
        relative_matrix = self.relative_values(sen_vector)
        clues_list = self.clueswords_values(sens_words)
        entities_list = self.entities_values(sens_words,sens_tag)
        options = self.generate_options(len(sens),num)
        max_value, best_option= 0,None
        tmp = []
        # self.save_value(Dir.res + "/FirstVersion/file"+str(self.count_index),text,coverage_list,relative_matrix,clues_list,entities_list)
        for option in options:
            option_score,tmp = self.score_option(option,coverage_list,relative_matrix,clues_list,entities_list)

            # tmp.append(str(option_score))
            if option_score > max_value:
                best_option = option
                max_value = option_score
                # print(best_option,max_value,tmp)
        # print(best_option)
        # for i in range(len(sens)):
        #     print(i,sens[i])
        # print(best_option)
        abstract = [sens[var] for var in best_option]
        # print('\n'.join(tmp),max_value)
        return abstract

def load_file(filepath):
    def filter(sen):
        return sen.strip()

    tmp = ftools.read_lines(filepath)
    return "。".join(map(filter, tmp))

if __name__=="__main__":
    # test_file = Dir.res+"/cleandata_604/news/training_4.txt"
    # text = ftools.read_lines(test_file)
    name = "trainning_2788.txt"
    text_path = Dir.res + "/cleandata_small/news/" + name
    text = load_file(text_path)
    summ = Summarizor()
    summ.set_weight([1,1,1,1])

    # print()

    print(summ.info)
    res = summ.summarize(text)
    for line in res:
        print(line)
