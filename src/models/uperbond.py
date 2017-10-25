import Dir
from src.tools import  FileTools as ftools
from src.tools import Tools as tools

class Uper():

    def __init__(self):
        self.name = "uper"
        self.info = self.name+" not num"
        self.answer = {}
        self.load()
    def load(self):
        path = Dir.res+"/cleandata_highquality_1640/abstract/"
        for name in ftools.get_files(path):
            tmp = ftools.read_lines(path+name)
            self.answer[name] = []
            for var in tmp:
                if len(var.strip()) <= 5 :
                    continue
                self.answer[name].append(var)

    def sim(self,sen1,sen2):
        count = 0
        words1 = tools.seperate(sen1)
        words2 = tools.seperate(sen2)
        for var in words1:
            if var in words2:
                count+=1
        if len(words1) ==0 :
            print("sentences",sen1)
            input()
        count /= len(words1)
        return count

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


    def summarize(self,text, num,fname):
        fname = fname + ".txt"
        res = []
        for i in range(len(self.answer[fname])):
            tmp = []
            for j in range(len(text)):
                tmp.append(self.sim(self.answer[fname][i],text[j]))
            # print(tmp)
            res.append(text[tmp.index(max(tmp))])
        return res

def generate_new_data():
    npath = Dir.res+"/cleandata_highquality_3500/news/"
    # apath = Dir.res+"/cleandata_highquality_3500/abstract/"

    new_npath = Dir.res+"/cleandata_highquality_3500_new/news/"
    new_apath = Dir.res+"/cleandata_highquality_3500_new/abstract/"

    uper = Uper()

    for name in ftools.get_files(npath):
        path = npath+name
        content = ftools.read_lines(path)
        new_abstract = uper.summarize(content,num =3,fname=name[:-4])
        ftools.copy(path,new_npath+name)
        ftools.write_list(new_apath+name,new_abstract)

if __name__ == "__main__":
    generate_new_data()
    # path = Dir.res+"/cleandata_highquality_3500/news/trainning_4.txt"
    # apath = Dir.res+"/cleandata_highquality_3500/abstract/trainning_4.txt"
    # text = ftools.read_lines(path)
    # summ = Uper()
    # abst = summ.summarize(text,3,"trainning_4.txt")
    # rabst = ftools.read_lines(apath)
    # for ra in rabst:
    #     print(ra)
    # for a in abst:
    #     print(a)