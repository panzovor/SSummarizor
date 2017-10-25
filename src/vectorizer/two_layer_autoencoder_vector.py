from __future__ import division, print_function, absolute_import
import time
import tensorflow as tf
import hashlib
import Dir
from src.tools import Tools as tools
from src.tools import FileTools as ftools
import numpy as np

class FastCoder():

    def __init__(self):
        self.auto = AutoCoder()
        self.name = "fast encoder"
        self.data ={}
        path = Dir.res+"/encoder/cleandata_8700/"
        fllist = ftools.get_files(path)
        for name in fllist:
            self.data[name] = tools.load_object(path+name)
        # print(len(self.data.keys()))

    def vectorize(self,sens_words,sens_tags):
        key_text = ''.join([''.join(var) for var in sens_words])
        key = tools.md5(key_text)
        # print(key)
        if key in self.data.keys():
            tmp = self.data[key]
        else:
            print("trainning")
            tmp0,tmp1 = self.auto.vectorize(sens_words,sens_tags)
            tmp = [tmp0,tmp1]
            tmpsens = [list(var) for var in tmp0]

            save_object = [tmpsens, list(tmp1)]
            save_key = tools.md5(key_text)
            tools.save_object(save_object, Dir.res+"/encoder/cleandata_8700/" + save_key)
        # print(type(tmp))
        # print(len(tmp))
        return tmp[0],tmp[1]


class AutoCoder():

    def __init__(self):
        self.name = "auto encoder"
        self.words = None
        self.data = None
        self.n_input = 200
        self.layer1_per_node = 16
        self.n_hidden_1 = 100
        self.n_hidden_2 = 16
        self.seed = 1
        self.repeat_times = 20

        self.dropout = 0.5

        self.sens_similarity = None
        self.train_data = []
        self.layer1_jihuo = tf.nn.softmax
        self.layer2_jihuo = tf.nn.sigmoid
        self.Xn = None
        self.Xv = None
        self.Xm = None
        self.Xo = None
        self.words_count =[]

        self.weights = None
        self.biases = None

        # Parameters
        self.learning_rate = 0.05
        self.display_step = 1
        self.examples_to_show = 10

        # Network Parameters

        self.encoder_op = None
        self.decoder_op = None

        self.cost = None
        self.optimizer = None
        self.now_epoch = 0

    def preprocess(self,text):
        sens_words, sens_tag = [], []
        sens = tools.seperate_sentences(text)
        for sen in sens:
            tmp_words, tmp_tag = [], []
            for w, t in tools.sen_pog(sen):
                tmp_words.append(w)
                tmp_tag.append(t)
            sens_words.append(tmp_words)
            sens_tag.append(tmp_tag)
        return sens, sens_words, sens_tag

    def set_parameter(self,itimes):
        self.repeat_times = itimes

    def set(self):
        # self.n_input = len(self.words[0])
        # self.n_hidden_1 = int(self.n_input / 2)
        # self.n_hidden_2 = 20


        n_input = len(self.words[0])
        v_input = len(self.words[1])
        m_input = len(self.words[2])
        o_input = len(self.words[3])

        self.Xn = tf.placeholder("float", [None, n_input])
        self.Xv = tf.placeholder("float", [None, v_input])
        self.Xm = tf.placeholder("float", [None, m_input])
        self.Xo = tf.placeholder("float", [None, o_input])
        self.weights = {
            'encoder_h1_n': tf.Variable(tf.random_normal([n_input, self.layer1_per_node],seed=self.seed)),
            'encoder_h1_v': tf.Variable(tf.random_normal([v_input, self.layer1_per_node],seed=self.seed)),
            'encoder_h1_m': tf.Variable(tf.random_normal([m_input, self.layer1_per_node],seed=self.seed)),
            'encoder_h1_o': tf.Variable(tf.random_normal([o_input, self.layer1_per_node],seed=self.seed)),
            'encoder_h2': tf.Variable(tf.random_normal([self.layer1_per_node * 4, self.n_hidden_2],seed=self.seed)),

            'decoder_h1_n': tf.Variable(tf.random_normal([self.n_hidden_2, self.layer1_per_node],seed=self.seed)),
            'decoder_h1_v': tf.Variable(tf.random_normal([self.n_hidden_2, self.layer1_per_node],seed=self.seed)),
            'decoder_h1_m': tf.Variable(tf.random_normal([self.n_hidden_2, self.layer1_per_node],seed=self.seed)),
            'decoder_h1_o': tf.Variable(tf.random_normal([self.n_hidden_2, self.layer1_per_node],seed=self.seed)),

            'decoder_h2_n': tf.Variable(tf.random_normal([self.layer1_per_node, n_input],seed=self.seed)),
            'decoder_h2_v': tf.Variable(tf.random_normal([self.layer1_per_node, v_input],seed=self.seed)),
            'decoder_h2_m': tf.Variable(tf.random_normal([self.layer1_per_node, m_input],seed=self.seed)),
            'decoder_h2_o': tf.Variable(tf.random_normal([self.layer1_per_node, o_input],seed=self.seed))

        }
        self.biases = {
            'encoder_b1_n': tf.Variable(tf.random_normal([self.layer1_per_node],seed=self.seed)),
            'encoder_b1_v': tf.Variable(tf.random_normal([self.layer1_per_node],seed=self.seed)),
            'encoder_b1_m': tf.Variable(tf.random_normal([self.layer1_per_node],seed=self.seed)),
            'encoder_b1_o': tf.Variable(tf.random_normal([self.layer1_per_node],seed=self.seed)),
            'encoder_b2': tf.Variable(tf.random_normal([self.n_hidden_2],seed=self.seed)),

            'decoder_b1_n': tf.Variable(tf.random_normal([self.layer1_per_node],seed=self.seed)),
            'decoder_b1_v': tf.Variable(tf.random_normal([self.layer1_per_node],seed=self.seed)),
            'decoder_b1_m': tf.Variable(tf.random_normal([self.layer1_per_node],seed=self.seed)),
            'decoder_b1_o': tf.Variable(tf.random_normal([self.layer1_per_node],seed=self.seed)),

            'decoder_b2_n': tf.Variable(tf.random_normal([n_input])),
            'decoder_b2_v': tf.Variable(tf.random_normal([v_input])),
            'decoder_b2_m': tf.Variable(tf.random_normal([m_input])),
            'decoder_b2_o': tf.Variable(tf.random_normal([o_input]))

        }
        # Parameters
        self.learning_rate = 0.05

        self.display_step = 1
        self.examples_to_show = 10

        # Network Parameters

        self.encoder_op = self.encoder(self.Xn,self.Xv,self.Xm,self.Xo)
        dexn,dexv,dexm,dexo = self.decoder(self.encoder_op)

        input_x = tf.concat([self.Xn,self.Xv,self.Xm,self.Xo],axis=1)
        output_x = tf.concat([dexn,dexv,dexm,dexo],axis=1)
        self.cost = tf.reduce_mean(tf.pow(input_x - output_x, 2))
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.cost)
        self.now_epoch = 0

        # Building the encoder
    def encoder(self,xn,xv,xm,xo):
        # Encoder Hidden layer with sigmoid activation #1
        # print(self.weights['encoder_h1_n'].shape)
        layer_1_n = tf.nn.dropout(self.layer1_jihuo(tf.add(tf.matmul(xn, self.weights['encoder_h1_n']),self.biases['encoder_b1_n'])),self.dropout,seed= self.seed)
                                      #,self.dropout)
        layer_1_v = tf.nn.dropout(self.layer1_jihuo(tf.add(tf.matmul(xv, self.weights['encoder_h1_v']),self.biases['encoder_b1_v'])),self.dropout,seed= self.seed)
                                      #,self.dropout)
        layer_1_m = tf.nn.dropout(self.layer1_jihuo(tf.add(tf.matmul(xm, self.weights['encoder_h1_m']),self.biases['encoder_b1_m'])),self.dropout,seed= self.seed)
                                      #,self.dropout)
        layer_1_o = tf.nn.dropout(self.layer1_jihuo(tf.add(tf.matmul(xo, self.weights['encoder_h1_o']),self.biases['encoder_b1_o'])),self.dropout,seed= self.seed)
                                      #,self.dropout)

        layer_1 = tf.concat([layer_1_n,layer_1_v,layer_1_m,layer_1_o],1)

        # Decoder Hidden layer with sigmoid activation #2
        layer_2 = self.layer2_jihuo(tf.add(tf.matmul(layer_1, self.weights['encoder_h2']),
                                       self.biases['encoder_b2']))

        # tf.

        return layer_2
    # Building the decoder
    def decoder(self,x):
        # Encoder Hidden layer with sigmoid activation #1
        layer_1_n = self.layer2_jihuo(tf.add(tf.matmul(x, self.weights['decoder_h1_n']), self.biases['decoder_b1_n']))
        layer_1_v = self.layer2_jihuo(tf.add(tf.matmul(x, self.weights['decoder_h1_v']), self.biases['decoder_b1_v']))
        layer_1_m = self.layer2_jihuo(tf.add(tf.matmul(x, self.weights['decoder_h1_m']), self.biases['decoder_b1_m']))
        layer_1_o = self.layer2_jihuo(tf.add(tf.matmul(x, self.weights['decoder_h1_o']), self.biases['decoder_b1_o']))

        layer_2_n = self.layer1_jihuo(tf.add(tf.matmul(layer_1_n, self.weights['decoder_h2_n']),self.biases['decoder_b2_n']))
        layer_2_v = self.layer1_jihuo(tf.add(tf.matmul(layer_1_v, self.weights['decoder_h2_v']),self.biases['decoder_b2_v']))
        layer_2_m = self.layer1_jihuo(tf.add(tf.matmul(layer_1_m, self.weights['decoder_h2_m']),self.biases['decoder_b2_m']))
        layer_2_o = self.layer1_jihuo(tf.add(tf.matmul(layer_1_o, self.weights['decoder_h2_o']),self.biases['decoder_b2_o']))
        # Decoder Hidden layer with sigmoid activation #2

        return layer_2_n,layer_2_v,layer_2_m,layer_2_o

    ### input: data
    ### output: dict{
    ### sen_index: noun_similarity[index_0,index_1,..., index_n-1,index_n]
    ###            verb_similarity[index_0,index_1,..., index_n-1,index_n]
    ###            nume_similarity[index_0,index_1,..., index_n-1,index_n]
    ### }
    def build_traindata(self):
        def simi(data1,data2,compare_index1,compare_index2):
            count =0
            for i in range(len(data1[compare_index1])):
                if data1[compare_index1][i] == data2[compare_index1][i] and data2[compare_index1][i] ==1:
                    count += 1
            for i in range(len(data1[compare_index2])):
                if data1[compare_index2][i] == data2[compare_index2][i] and data2[compare_index2][i] ==1:
                    count += 1
            return count/(len(data1[compare_index1])+len(data1[compare_index2]))
        def combine(datas):
            new_data = []
            for i in range(len(datas[0])):
                tmp =[]
                for j in range(len(datas[0][i])):
                    tmp.append(max([var[i][j]  for var in  datas]))
                new_data.append(tmp)
            return new_data

        self.sens_similarity = {}
        for i in range(len(self.data)):
            self.sens_similarity[i] = [{},{},{}]
            for j in range(len(self.data)):
                if i!=j:
                    self.sens_similarity[i][0][j] = simi(self.data[i],self.data[j],1,2)
                    self.sens_similarity[i][1][j] = simi(self.data[i],self.data[j],0,2)
                    self.sens_similarity[i][2][j] = simi(self.data[i],self.data[j],0,1)
        for key in self.sens_similarity.keys():
            self.sens_similarity[key][0] = list(sorted(self.sens_similarity[key][0].items(),key = lambda d:d[1],reverse=True))
            self.sens_similarity[key][1] = list(sorted(self.sens_similarity[key][1].items(),key = lambda d:d[1],reverse=True))
            self.sens_similarity[key][2] = list(sorted(self.sens_similarity[key][2].items(),key = lambda d:d[1],reverse=True))

        nd,vd,md,od =[],[],[],[]

        for k in range(3):
            for i in range(len(self.data)):
                tmp = []
                # print(self.data[self.sens_similarity[i][1][0]])
                # print(self.data[self.sens_similarity[i][2][0]])
                tmp.append(self.data[self.sens_similarity[i][0][k][0]])
                tmp.append(self.data[self.sens_similarity[i][1][k][0]])
                tmp.append(self.data[self.sens_similarity[i][2][k][0]])
                tmp_data = combine(tmp)
                nd.append(tmp_data[0])
                vd.append(tmp_data[1])
                md.append(tmp_data[2])
                od.append(tmp_data[3])
        self.train_data = [nd,vd,md,od]

    def process(self,sens_words,sens_tag):
        words_n = []
        words_v = []
        words_m = []
        words_other = []
        count_n,count_v,count_m,count_o = [],[],[],[]
        for i in range(len(sens_words)):
            sen = sens_words[i]
            tag = sens_tag[i]
            for j in range(len(sen)):
                if tag[j][0] == "n":
                    if sen[j] not in words_n:
                        words_n.append(sen[j])
                        count_n.append(0)
                    count_n[words_n.index(sen[j])]+=1
                elif tag[j][0] == "v":
                    if sen[j] not in words_v:
                        words_v.append(sen[j])
                        count_v.append(0)
                    count_v[words_v.index(sen[j])] += 1
                elif tag[j][0] == "m":
                    if sen[j] not in words_m:
                        words_m.append(sen[j])
                        count_m.append(0)
                    count_m[words_m.index(sen[j])] += 1
                else:
                    if sen[j] not in words_other:
                        words_other.append(sen[j])
                        count_o.append(0)
                    count_o[words_other.index(sen[j])] += 1
        words = [words_n,words_v,words_m,words_other]
        vector = []
        for sen in sens_words:
            tmp_n,tmp_v,tmp_m,tmp_other =[0]*len(words_n),[0]*len(words_v),[0]*len(words_m),[0]*len(words_other)
            for w in sen:
                if w in words_n:
                    tmp_n[words_n.index(w)] =1
                elif w in words_v:
                    tmp_v[words_v.index(w)] =1
                elif w in words_m:
                    tmp_m[words_m.index(w)] =1
                elif w in words_other:
                    tmp_other[words_other.index(w)] =1
            vector.append([tmp_n,tmp_v,tmp_m,tmp_other])
        return words,vector,[count_n,count_v,count_m,count_o]

    def vectorize(self,sens_words,sens_tag = None):

        tf.reset_default_graph()
        self.words, self.data,self.words_count = self.process(sens_words,sens_tag)
        self.set()
        tmp = self.trainWithData()
        return tmp[:-1],tmp[-1]

    def get_sens_data(self):
        nd,vd,md,od =[],[],[],[]
        for i in range(len(self.data)):
            nd.append(self.data[i][0])
            vd.append(self.data[i][1])
            md.append(self.data[i][2])
            od.append(self.data[i][3])

        nd.append([1]*len(self.words[0]))
        vd.append([1]*len(self.words[1]))
        md.append([1]*len(self.words[2]))
        od.append([1]*len(self.words[3]))
        return nd,vd,md,od

    def trainWithData(self):
        self.build_traindata()
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            for j in range(self.repeat_times):
                nd,vd,md,od= self.train_data[0],self.train_data[1],self.train_data[2],self.train_data[3]
                _,cost = sess.run([self.optimizer,self.cost],feed_dict={self.Xn:nd, self.Xv:vd, self.Xm:md, self.Xo:od})
                # print("Epoch:", '%04d' % (j+1),
                #       "cost=", "{:.9f}".format(cost))
            nd,vd,md,od = self.get_sens_data()
            encoder = sess.run(self.encoder_op, feed_dict={self.Xn:nd, self.Xv:vd, self.Xm:md, self.Xo:od})
            return encoder

    def generate_data(self,text):
        if isinstance(text,str):
            sens = tools.seperate_sentences(text)
        else:
            sens = text
        words =[]
        sen_words = []
        for sen in sens:
            wp = tools.sen_pog( sen)
            tmp = []
            for w,p in wp:
                if "n" in p or "v" in p or "m" in p:
                    tmp.append(w)
                    if w not in words:
                        words.append(w)
            sen_words.append(tmp)
        vector = []
        for sen_w in sen_words:
            tmp =[0]*len(words)
            for i in range(len(words)):
                w=  words[i]
                if w in sen_w:
                    tmp[i] = 1
            vector.append(tmp)
        return words,vector

def load_file(filepath):
    def filter(sen):
        return sen.strip()

    tmp = ftools.read_lines(filepath)
    return "。".join(map(filter, tmp))

def vectorize_files(fileroot,savepath):
    data = ftools.read_dir_lines_dict(fileroot)
    auto  = AutoCoder()
    count = 0
    print(len(data.keys()))
    for key in data.keys():

        text = '。'.join(data[key])

        sens, sens_words, sens_tags = auto.preprocess(text)
        start = time.time()
        sens_vector,essay_vector = auto.vectorize(sens_words, sens_tags)
        end = time.time()
        key_text =''.join([''.join(var) for var in sens_words])

        save_key = tools.md5(key_text)
        tmp =[list(var) for var in sens_vector]

        save_object = [tmp,list(essay_vector)]

        tools.save_object(save_object,Dir.res + "/encoder/cleandata_8700/"+save_key)

        count+=1

        print(count,len(data.keys()),end-start)

def load_vectorize_files(vectorize_path):
    lines = ftools.read_lines(vectorize_path)
    res = {}
    for line in lines:
        seperate_point = line.rindex("\t")
        key = line[:seperate_point]
        content = seperate_point[seperate_point+1:][2:-2]
        vectors = [float(var) for var in content.split("','")]
        if key not in res.keys():
            res[key] = vectors
    return res



if __name__ == "__main__":
    text_path = Dir.res + "/cleandata_small/news/trainning_4.txt"
    text = load_file(text_path)
    auto = AutoCoder()
    sens,sens_words,sens_tags = auto.preprocess(text)
    tmp = auto.vectorize(sens_words,sens_tags)
    # print(len(sens),len(tmp))
    print(len(tmp[0][0]))
    print(len(tmp[1]))
    print(tmp[0][0],tmp[1])
    from src.models.SecondVersion import Dist as Dist
    from src.models.SecondVersion import Distance as Distance
    dist = Dist()

    for i in range(len(tmp[0])):
        print(sens[i],",",dist.sim(tmp[0][i],tmp[1],dis=Distance.COS))
