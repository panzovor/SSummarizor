import tensorflow as tf
from src.tools import Tools as tools
from src.tools.Tools import Dist as Dist
from src.tools.Tools import Distance as Distance
from src.tools import FileTools as ftools
import Dir
import time


class Auto_Simple_Vec():

    def __init__(self):
        self.name = "simple auto encoder"
        self.words_bag = {}
        self.middle_layer_size = 20
        self.learning_rate = 1
        self.repeat_time = 20
        self.cost_threshold = 2
        self.batch_size = 0.1
        self.now_index = 0
        self.dist = Dist()
        self.words_tf = {}
        self.min_count = 2

        #network parameter
        self.en_para = None
        self.de_para = None
        self.xl = None
        self.xr = None
        self.y = None
        self.weight = {
            "encoder": None,
            "decoder": None
        }

        self.bias = {
            "encoder": None,
            "decoder": None
        }
        self.encoder_op =None
        self.decoder_op =None
        self.activate = tf.tanh
        self.cost = None

    def set_parameter(self,iter_times):
        self.repeat_time = iter_times

    def preprocess(self,text):
        sens_words, sens_tag = [], []
        sens = tools.seperate_sentences(text)
        tmp= []
        for i in range(1,len(sens)):
            sen = sens[i]
        # for sen in sens:
            if "原标题" in sen :
                continue
            tmp.append(sen)
            tmp_words,tmp_tag = tools.seperate_pog(sen)
            sens_words.append(tmp_words)
            sens_tag.append(tmp_tag)

        return tmp, sens_words, sens_tag

    def build_words_bag(self,sens_words):
        words_bag = {}
        self.middle_layer_size = len(sens_words)
        for var in sens_words:
            for w in var:
                if w not in self.words_tf:
                    self.words_tf[w] = 0
                self.words_tf[w] +=1
        for i in range(len(sens_words)):
            for j in range(len(sens_words[i])):
                w = sens_words[i][j]
                if w not in words_bag and self.words_tf[w] >= self.min_count:
                    words_bag[w] = len(words_bag)
        sum_tf = sum([self.words_tf[var] for var in self.words_tf.keys()])

        self.en_para = [[0]*self.middle_layer_size for var in range(len(words_bag))]
        self.de_para = [[0] * len(words_bag) for var in range(self.middle_layer_size)]



        for w in words_bag.keys():
            for i in range(self.middle_layer_size):
                tmp =0.2
                if w in sens_words[i] :
                    tmp =0.8
                self.en_para[words_bag[w]][i] = self.words_tf[w]*tmp/ sum_tf
                self.de_para[i][words_bag[w]]  = self.words_tf[w]*tmp/ sum_tf
        return words_bag
        # print("length_words_bag",len(self.words_bag))

    def words2worvect(self,words,words_bag):
        tmp = [0] * len(words_bag)
        for w in words:
            if w in words_bag.keys():
                tmp[words_bag[w]] =1
        return tmp

    def generate_train_data(self,sens_words,words_bag):
        sens_vec = []
        train_data=[]
        for i in range(len(sens_words)):
            sens_vec.append(self.words2worvect(sens_words[i],words_bag))
        # print("sens_vec length",len(sens_vec))
        for i in range(len(sens_vec)):
            for j in range(len(sens_vec)):
                if i != j:
                    train_data.append([sens_vec[i],sens_vec[j],[self.dist.sim(sens_vec[i],sens_vec[j],Distance.EUD)]])
        return train_data

    def init_network(self,words_bag):
        self.xl = tf.placeholder("float", [None, len(words_bag)])
        self.xr = tf.placeholder("float", [None, len(words_bag)])
        self.y = tf.placeholder("float", [None, 1])



        self.weight["encoder"] = tf.Variable(tf.convert_to_tensor(self.en_para))
        self.weight["decoder"] = tf.Variable(tf.convert_to_tensor(self.de_para))
        self.bias["encoder"] = tf.Variable(tf.convert_to_tensor([0.5]*self.middle_layer_size))
        self.bias["decoder"] = tf.Variable(tf.convert_to_tensor([0.5]*len(words_bag)))

        # self.weight["encoder"] = tf.Variable(tf.random_normal([len(self.words_bag),self.middle_layer_size]))
        # self.weight["decoder"] = tf.Variable(tf.random_normal([self.middle_layer_size,len(self.words_bag)]))
        # self.bias["encoder"] = tf.Variable(tf.random_normal([self.middle_layer_size]))
        # self.bias["decoder"] = tf.Variable(tf.random_normal([len(self.words_bag)]))


        self.encoder_op = self.activate(tf.add(tf.matmul(self.xl,self.weight["encoder"]),self.bias["encoder"]))
        encoder_opl = self.activate(tf.add(tf.matmul(self.xl,self.weight["encoder"]),self.bias["encoder"]))
        encoder_opr = self.activate(tf.add(tf.matmul(self.xr,self.weight["encoder"]),self.bias["encoder"]))
        decoder_opl = self.activate(self.activate(tf.add(tf.matmul(encoder_opl,self.weight["decoder"]),self.bias["decoder"])))
        decoder_opr = self.activate(tf.add(tf.matmul(encoder_opr,self.weight["decoder"]),self.bias["decoder"]))

        costl = tf.pow(decoder_opl-self.xl,2)
        costr = tf.pow(decoder_opr-self.xr,2)

        # costl = tf.nn.softmax_cross_entropy_with_logits(decoder_opl,self.xl)
        # costr = tf.nn.softmax_cross_entropy_with_logits(decoder_opr,self.xr)

        x3_norm = tf.sqrt(tf.reduce_sum(tf.square(encoder_opl)))
        x4_norm = tf.sqrt(tf.reduce_sum(tf.square(encoder_opr)))
        x3_x4 = tf.reduce_sum(tf.multiply(encoder_opl, encoder_opr))
        cosin = tf.div(x3_x4 ,(x3_norm * x4_norm))

        # eud = tf.sqrt(tf.reduce_sum(tf.pow(encoder_opl - encoder_opr, 2)))
        #
        self.cost = tf.reduce_mean(tf.add(tf.add(costl,costr),8*tf.sqrt(tf.pow((self.y-cosin),2))))
        # self.cost = tf.reduce_mean(tf.add(tf.add(costl,costr),eud))
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.cost)
        return cosin


    # def next_batch(self):
    #     tmp1,tmp2,tmp3 = [],[],[]
    #
    #     size = int(len(self.train_data)*self.batch_size)
    #     for i in range(size):
    #         if self.now_index >= len(self.train_data):
    #             self.now_index = 0
    #         tmp1.append(self.train_data[self.now_index][0])
    #         tmp2.append(self.train_data[self.now_index][1])
    #         tmp3.append(self.train_data[self.now_index][2])
    #         self.now_index+=1
    #     return tmp1,tmp2,tmp3

    def vectorize(self,sens_words,sens_tags = None):
        # start = time.time()
        tf.reset_default_graph()
        words_bag = self.build_words_bag(sens_words)
        train_data = self.generate_train_data(sens_words,words_bag)
        self.init_network(words_bag)
        # print("network initial")
        # self.generate_train_data(sens_words)
        init = tf.global_variables_initializer()
        sens_vec,essay_vec =[],None
        # print("strat training")
        with tf.Session() as sess:
            sess.run(init)
            axl, axr, ay = [], [], []
            for i in range(len(train_data)):
                axl.append(train_data[i][0])
                axr.append(train_data[i][1])
                ay.append(train_data[i][2])
            i,cost = 0,10
            while i<self.repeat_time and ( self.cost_threshold<cost or True):
                _, cost = sess.run([self.optimizer,self.cost],feed_dict={self.xl: axl,self.xr:axr,self.y:ay})
                i+=1
                # print(i,cost)

            for i in range(len(sens_words)):
                sens_i_vec = sess.run(self.encoder_op,feed_dict={self.xl:[self.words2worvect(sens_words[i],words_bag)]})
                sens_vec.append(list(sens_i_vec)[0])
            essay_vec= list(sess.run(self.encoder_op,feed_dict={self.xl:[[1]*len(words_bag)]})[0])
        # endtime = time.time()
        # print(endtime-start)
        return sens_vec,essay_vec


if __name__ == "__main__":
    path = Dir.res+"/cleandata_small/news/trainning_2788.txt"
    text = ftools.read_lines(path)
    text = '。'.join(text)
    asv = Auto_Simple_Vec()
    sens,sens_words,sens_tags = asv.preprocess(text)

    # for var in sens_words:
    #     print(var)

    print("se_words lgth" , len(sens_words))
    sen_vec,essay_vec = asv.vectorize(sens_words,sens_tags)
    # print(essay_vec)
    print(sens[0],sens[1])
    print(asv.dist.sim(sen_vec[0],sen_vec[1]))
    print(asv.dist.sim(sen_vec[0],sen_vec[-1]))
    print(asv.dist.sim(sen_vec[2],sen_vec[3]))

    coverage_list =[]
    for i in range(len(sen_vec)):
        # print(sen_vec[i])
        # input()
        coverage_list.append(asv.dist.sim(sen_vec[i],essay_vec,Distance.EUD))
    tools.normarlize(coverage_list)
    for i in range(len(coverage_list)):
        print(sens[i],",",coverage_list[i])


    # print("------en--------")
    # print(essay_vec)