import numpy as np
import tensorflow as tf
import Dir
from src.tools import FileTools as ftools
from src.tools import Tools as tools
from src.vectorizer import words_bag_vector as wbv


class lstm_vector():

    def __init__(self):
        self.name = "lstm_vector"
        ### training parameter
        self.train_times =100
        self.batch_size = None
        self.lr = 0.1

        ### lstm structure parameter
        self.hidden_size = 50
        self.x = None
        self.num_step = None
        self.num_layers = 1
        self.drop_out_rate = None
        self.data_type = tf.float32
        self.learning_rate = 0.05
        self.essay_vector = None
        self.output= None
        self.optimizer = None
        self.wordbag_vector = wbv.words_bag_vector()
        self.now_epoch = 0
        self.noise_rate = 0.2
        self.data =[]
        self.words_tags_dict = {}
        self.cost = None

    def set(self,sens_words):
        num_steps = len(sens_words)
        self.genenrate_data(sens_words)
        self.set_parameter(num_step=num_steps,input_size=len(self.wordbag_vector.words_bag))



    def set_parameter(self,num_step,input_size,hidden_nodes =20, num_layers =1,drop_out_rate=0,batch_size = 1):
        self.hidden_size= hidden_nodes
        self.num_step = num_step
        self.batch_size = batch_size
        self.x = tf.placeholder(dtype=self.data_type,shape = [None,self.num_step,input_size])
        self.num_layers = num_layers
        self.drop_out_rate = drop_out_rate

        self.sen_weight = tf.Variable(tf.random_normal(shape = [self.num_step,self.hidden_size]))
        self.sen_bias = tf.Variable(tf.random_normal(shape=[self.num_step]))

        self.essay_vector = None
        self.output = None
        self.optimizer = None
        self.network_structure()


    def network_structure(self):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size, forget_bias=1.0, state_is_tuple=True)

        # lstm_cell = tf.nn.dropout(lstm_cell,keep_prob=1-self.drop_out_rate)
        init_state = lstm_cell.zero_state(batch_size=self.batch_size,dtype=self.data_type)

        self.output, states= tf.nn.dynamic_rnn(lstm_cell, self.x, initial_state=init_state,dtype=self.data_type)
        # print(self.output.shape,states.shape)
        states is a tuple

        # self.output, states = lstm_cell(self.x, state)
        output = tf.reshape(self.output,[self.batch_size,-1])
        states_list= tf.reshape(states,shape=[self.batch_size,self.num_step,-1]).eval()
        self.essay_vector= []
        # a tensor  = tf.convert_to_tensor(array)
        # a numpy array = tensor.eval()

        # tensor_state =
        for batch_i in range(len(states_list )):
            tmp =[states_list[batch_i][0]]
            for num_step_j in range(len(states_list[batch_i])-1):
                tmp.append(states_list[batch_i][num_step_j+1]- states_list[batch_i][num_step_j])
            self.essay_vector.append(tf.nn.sigmoid(tf.add(tf.matmul(tmp,self.sen_weight),self.sen_bias)).eval())
        self.essay_vector = tf.convert_to_tensor(self.essay_vector)
        self.cost = tf.reduce_mean(tf.pow(output-self.essay_vector),2)
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.cost)

    def get_sens_words(self,text):
        sens = tools.seperate_sentences(text)
        sens_words = []
        for line in sens:
            words, tags = tools.seperate_pog(line)
            for i in range(len(words)):
                w = words[i]
                if w not in self.words_tags_dict.keys():
                    self.words_tags_dict[w] = tags[i]
            sens_words.append(words)
        return sens_words

    def genenrate_data(self,sens_words):
        sens_Vec,essay_vect = self.wordbag_vector.vectorize(sens_words)
        # print(sens_Vec[0])
        self.data = np.array([sens_Vec])
        tagbag = []
        for w in self.wordbag_vector.words_bag:
            tagbag.append(self.words_tags_dict[w])
        return sens_Vec,tagbag

    def next_epoch(self):
        return self.data

    def train(self):
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            for i in range(self.train_times):
                _, cost = sess.run([self.optimizer, self.cost], feed_dict={self.x: self.data})
                print("Epoch:", '%04d' % (i+1),
                          "cost=", "{:.9f}".format(cost))


def load_file(filepath):
    def filter(sen):
        return sen.strip()

    tmp = ftools.read_lines(filepath)
    return "。".join(map(filter, tmp))

if __name__ == "__main__":
    lv = lstm_vector()
    name = "training_4.txt"
    text_path = Dir.res + "/cleandata_604/news/" + name
    text = load_file(text_path)
    sens_words = lv.get_sens_words(text)
    lv.set(sens_words)
    # lv.train()