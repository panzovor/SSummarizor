import gensim
from gensim.models.word2vec import Word2Vec
from src.tools import FileTools as ftools
from src.tools import Tools as tools
import Dir

def loaddata(path):
    # flist = ftools.get_files(data_root)

    # count =1
    # for name in flist:
    #     print(count,len(flist))
    #     count+=1
    #     path = data_root+name
    trainformat_sentences = []
    content  = ftools.read_lines(path)
    for line in content:
        article = line[line.rindex(",")+1:]
        sentences = tools.seperate_sentences(article)
        for sen in sentences:
            trainformat_sentences.append(tools.seperate(sen))
    return trainformat_sentences

def train(traindata,savepath = Dir.res+"/parameter/words_vector/w2v.model"):
    ftools.check_filename(savepath)
    model = Word2Vec(sentences=traindata,size=200,window=5,min_count=3,workers=4)
    model.save(savepath)


def load(path= Dir.res+"/parameter/words_vector/w2v.model"):
    model = Word2Vec.load(path)
    return model

if __name__ == "__main__":
    root= Dir.res+"/data/"
    flist= ftools.get_files(root)
    data = []
    count = 0
    for name in flist:
        path = root+name
        print(" %04d" % count,len(flist))
        count+=1
        data.extend(loaddata(path))
    train(data)

