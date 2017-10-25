import tensorflow as tf
import src.tools.Tools as tools


class NNModel():

    def __init__(self):
        self.sentence_vector_size = 100
        self.sentence = []
        self.datatype = tf.float32


    def ed_network_structure(self,input_size):
        x = tf.placeholder(self.datatype)
        y = tf.placeholder(self.datatype)

        ### encoder

        w =  tf.Variable(tf.truncated_normal((input_size,self.sentence_vector_size)))
        b = tf.Variable(tf.zeros(self.sentence_vector_size))
        hid = tf.sigmoid(tf.add(tf.matmul(x,w),b))

        ### decoder
        wd = tf.Variable(tf.truncated_normal((self.sentence_vector_size,input_size)))
        bd = tf.Variable(tf.zeros(input_size))
        res = tf.sigmoid(tf.add(tf.matmul(hid,wd),bd))

        ### lost
        




    def ed_sentence(self,essay):
        self.sentence =tools.seperate_sentences(essay)

