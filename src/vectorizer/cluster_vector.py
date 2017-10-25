import random
import numpy as np
from numpy import linalg as LA
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

class SpectralClustering():

    '''
    cluster the input matrix ,return the nodes' label
    input : W: a 2-deminsion matrix
            k: the number of clusters
    output: the lables of nodes
    '''
    def cluster(self,W, k):
        """
        谱聚类
        :param points: 样本点
        :param k: 聚类个数
        :return: 聚类结果
        """
        # D = np.diag(np.sum(W, axis=1))
        # Dn = np.sqrt(LA.inv(D))
        # 本来应该像上面那样写，数学变换，写成了下面一行
        Dn = np.diag(np.power(np.sum(W, axis=1), -0.5))
        # 拉普拉斯矩阵：L=Dn*(D-W)*Dn=I-Dn*W*Dn
        # 也是做了数学变换的，简写为下面一行
        L = np.eye(len(W)) - np.dot(np.dot(Dn, W), Dn)
        eigvals, eigvecs = LA.eig(L)
        # 前k小的特征值对应的索引，argsort函数
        indices = np.argsort(eigvals)[:k]
        # 取出前k小的特征值对应的特征向量，并进行正则化
        k_smallest_eigenvectors = normalize(eigvecs[:, indices])
        # 利用KMeans进行聚类
        return KMeans(n_clusters=k).fit_predict(k_smallest_eigenvectors)


class MyVector():

    def __init__(self):
        self.name = "my vector"
        self.cluster = SpectralClustering()

    def kmeans(self,data,k=3,random_start = True):
        npdata = np.array(data)
        rs = 0
        if random_start:
            rs = random.Random().randint(0,len(data))
            kmeans = KMeans(n_clusters=k, random_state=rs,n_init=20).fit(npdata)
        else:
            init  =[]
            per = int(len(data)/k)
            for i in range(len(data)):
                if len(init) == k:
                    break
                if i % per == 0:
                    init.append(data[i])
            init = np.array(init)
            kmeans = KMeans(n_clusters=k,init = init,n_init = 1, random_state=rs).fit(npdata)

        clusters = list(kmeans.labels_)
        return  clusters

    def build_graph(self,sens_words,sens_tags):
        words_bag = []
        index_words = {}
        words_tag = {}
        for i in range(len(sens_words)):
            for j in range(len(sens_words[i])):
                if sens_words[i][j] not in words_bag:
                    words_bag.append(sens_words[i][j])
                    index_words[sens_words[i][j]] = len(index_words)
                    if sens_words[i][j] not in words_tag.keys():
                        words_tag[sens_words[i][j]] = sens_tags[i][j]

        matrix = [[0]* len(words_bag) for var in range(len(words_bag))]
        for i in range(len(sens_words)):
            for j in range(len(sens_words[i])-1):
                for k in range(j+1,len(sens_words[i])):
                    matrix[index_words[sens_words[i][j]]][index_words[sens_words[i][k]]] +=1
                    matrix[index_words[sens_words[i][k]]][index_words[sens_words[i][j]]] +=1

        return matrix,words_tag,words_bag,index_words


    def map_graph_single_role(self,graph,words_tag,index_words,role = "n"):
        def get_value(graph,start_index,end_index):
            value = 0
            for k in range(len(graph)):
                value+= graph[start_index][k]+graph[k][end_index]
            return value

        role_words = []
        role_index_words = {}
        for var in words_tag.keys():
            if role!= "o" and words_tag[var][0] == role:
                if var not in role_words:
                    role_words.append(var)
                    if var not in role_index_words.keys():
                        role_index_words[var] = len(role_index_words)
            elif role == "o" and words_tag[var][0] not in ["n","v"]:
                if var not in role_words:
                    role_words.append(var)
                    if var not in role_index_words.keys():
                        role_index_words[var] = len(role_index_words)

        role_graph =[[0]* len(role_words) for var in range(len(role_words))]

        for i in range(len(role_words)-1):
            for j in range(i+1,len(role_words)):
                role_graph[i][j] = get_value(graph,index_words[role_words[i]],index_words[role_words[j]])
                role_graph[j][i] = role_graph[i][j]
        return role_graph,role_words,role_index_words

    def print_graph(self,graph):
        for var in graph:
            print(var)

    def vectorize(self,sens_words, sens_tags):

        normal_garph, words_tag, words_bag, index_words = self.build_graph(sens_words,sens_tags)
        noun_graph,noun_words,noun_index_words = self.map_graph_single_role(normal_garph,words_tag,index_words,role = "n")
        verb_graph,verb_words,verb_index_words = self.map_graph_single_role(normal_garph,words_tag,index_words,role = "v")
        other_graph,othre_words,other_index_words = self.map_graph_single_role(normal_garph,words_tag,index_words,role= "o")

        # self.print_graph(normal_garph)
        # print("-----------------------------")
        # self.print_graph(noun_graph)
        # print("-----------------------------")
        # self.print_graph(verb_graph)
        # print("-----------------------------")
        # self.print_graph(other_graph)
        # print("-----------------------------")

        ks = [5,5,10]
        labels_noun_graph = self.cluster.cluster(noun_graph,k = ks[0])
        labels_verb_graph = self.cluster.cluster(verb_graph,k = ks[1])
        labels_other_graph= self.cluster.cluster(other_graph,k = ks[2])

        sens_vectors,essay_vector = [],[0]*sum(ks)
        for i in range(len(sens_words)):
            tmp = [0]*sum(ks)
            for j in range(len(sens_words[i])):
                if sens_words[i][j] in noun_index_words.keys():
                    noun_index =noun_index_words[sens_words[i][j]]
                    tmp[labels_noun_graph[noun_index]] +=1
                    essay_vector[labels_noun_graph[noun_index]]+=1

                if sens_words[i][j] in verb_index_words.keys():
                    verb_index = verb_index_words[sens_words[i][j]]
                    tmp[labels_verb_graph[verb_index]+ks[0]] += 1
                    essay_vector[labels_verb_graph[verb_index]] += 1

                if sens_words[i][j] in other_index_words.keys():
                    other_index = other_index_words[sens_words[i][j]]
                    tmp[labels_other_graph[other_index]+ks[1]+ks[0]] += 1
                    essay_vector[labels_other_graph[other_index]] += 1

            sens_vectors.append(tmp)
        return sens_vectors,essay_vector

if __name__ == "__main__":

    # sen2v = Sen2Vec()
    # sen2v.train()
    # doc2v= Doc2Vec()
    # doc2v.train()
    from src.tools import FileTools as ftools
    from src.tools import Tools as tools
    import Dir

    sens = ftools.read_lines(Dir.res+"/cleandata_604/news/training_4.txt")
    myvec = MyVector()
    sens_words,sens_pog =[],[]
    for line in sens:
        w,p = tools.seperate_pog(line)
        sens_words.append(w)
        sens_pog.append(p)
    sens,essay = myvec.vectorize(sens_words,sens_pog)
    print(sens[0])
    for ss in sens:
        print(ss)
    print(essay)