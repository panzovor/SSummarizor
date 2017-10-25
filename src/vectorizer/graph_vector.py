from src.tools import FileTools as ftools
from src.tools import Tools as tools
import src.models.HITS as HITS


class Graph_Vec():

    def __init__(self):
        self.name = "graph vector"
        self.targets = ["n","v","m"]

    def build_graph(self,sens_words,sens_tag):
        noun_graph ,verb_graph,num_graph,oth_graph = {},{},{},{}
        noun,verb,num,other ={},{},{},{}
        label = "sen_"
        for i in range(len(sens_words)):
            for j in range(len(sens_words[i])):
                if "n" in sens_tag[i][j]:
                    if sens_words[i][j] not in noun:
                        noun[sens_words[i][j]] = len(noun)
                    if sens_words[i][j] not in noun_graph.keys():
                        noun_graph[sens_words[i][j]] = []
                    if label+str(i) not in noun_graph[sens_words[i][j]]:
                        noun_graph[sens_words[i][j]].append(label+str(i))
                    if label+str(i) not in noun_graph.keys():
                        noun_graph[label+str(i)] = []
                    if sens_words[i][j] not in noun_graph[label+str(i)]:
                        noun_graph[label+str(i)].append(sens_words[i][j])
                elif "v" in sens_tag[i][j]:
                    if sens_words[i][j] not in verb:
                        verb[sens_words[i][j]] = len(verb)
                    if sens_words[i][j] not in verb_graph.keys():
                        verb_graph[sens_words[i][j]] = []
                    if label+str(i) not in verb_graph[sens_words[i][j]]:
                        verb_graph[sens_words[i][j]].append(label+str(i))
                    if label+str(i) not in verb_graph.keys():
                        verb_graph[label+str(i)] = []
                    if sens_words[i][j] not in verb_graph[label+str(i)]:
                        verb_graph[label+str(i)].append(sens_words[i][j])

                elif "m" in sens_tag[i][j]:
                    if sens_words[i][j] not in num:
                        num[sens_words[i][j]] = len(num)
                    if sens_words[i][j] not in num_graph.keys():
                        num_graph[sens_words[i][j]] = []
                    if label+str(i) not in num_graph[sens_words[i][j]]:
                        num_graph[sens_words[i][j]].append(label+str(i))
                    if label+str(i) not in num_graph.keys():
                        num_graph[label+str(i)] = []
                    if sens_words[i][j] not in num_graph[label+str(i)]:
                        num_graph[label+str(i)].append(sens_words[i][j])

                else:
                    if sens_words[i][j] not in other:
                        other[sens_words[i][j]] = len(other)
                    if sens_words[i][j] not in oth_graph.keys():
                        oth_graph[sens_words[i][j]] = []
                    if label+str(i) not in oth_graph[sens_words[i][j]]:
                        oth_graph[sens_words[i][j]].append(label+str(i))
                    if label+str(i) not in oth_graph.keys():
                        oth_graph[label+str(i)] = []
                    if sens_words[i][j] not in oth_graph[label+str(i)]:
                        oth_graph[label+str(i)].append(sens_words[i][j])

        return noun_graph,verb_graph,num_graph,oth_graph,noun,verb,num,other

    def standlize(self,vec):
        vsum = sum(vec)
        if vsum>0:
            for i in range(len(vec)):
                vec[i] /=vsum

    def vectorize(self, sens_words, sens_tag):
        ngraph,vgraph,mgraph,oth_graph,noun,verb,num,other = self.build_graph(sens_words,sens_tag)

        nau,nhub=  HITS.HITS(ngraph)
        vau,vhub=  HITS.HITS(vgraph)
        mau,mhub=  HITS.HITS(mgraph)
        oau,ohub=  HITS.HITS(oth_graph)
        label = "sen_"
        sens_vecs = []
        for i in range(len(sens_words)):
            n_vec,v_vec,m_vec,o_vec = [0]*len(noun),[0]*len(verb),[0]*len(num),[0]*len(other)
            for j in range(len(sens_words[i])):
                # print(i)
                # print(sens_words[i])
                # print(j,sens_words[i][j])
                # print(num)
                # print("---------")
                if sens_words[i][j] in noun and label+str(i) in nhub.keys():
                    n_vec[noun[sens_words[i][j]]] = nau[sens_words[i][j]]*nhub[label+str(i)]/(sum([nhub[var] for var in ngraph[sens_words[i][j]]]))
                elif sens_words[i][j] in verb and label+str(i) in vhub.keys():
                    v_vec[verb[sens_words[i][j]]] = vau[sens_words[i][j]]*vhub[label+str(i)]/(sum([vau[var] for var in vgraph[sens_words[i][j]]]))
                elif sens_words[i][j] in num.keys() and label+str(i) in mhub.keys():
                    m_vec[num[sens_words[i][j]]] = mau[sens_words[i][j]]*mhub[label+str(i)]/(sum([mau[var] for var in mgraph[sens_words[i][j]]]))
                elif sens_words[i][j] in other and label+str(i) in ohub.keys():
                    o_vec[other[sens_words[i][j]]] = oau[sens_words[i][j]]*ohub[label+str(i)]/(sum([oau[var] for var in oth_graph[sens_words[i][j]]]))
            sens_vecs.append(n_vec+v_vec+m_vec+o_vec)

        en_vec,ev_vec,em_vec,eo_vec=[0]*len(noun),[0]*len(verb),[0]*len(num),[0]*len(other)
        for var in noun:
            en_vec[noun[var]] = nau[var]
        for var in verb:
            ev_vec[verb[var]] = vau[var]
        for var in num:
            em_vec[num[var]] = mau[var]
        for var in other:
            eo_vec[other[var]] = oau[var]
        essay_vector = en_vec+ev_vec+em_vec+eo_vec

        return sens_vecs,essay_vector

import Dir
if __name__ == "__main__":
    name = "training_4.txt"
    text_path = Dir.res + "/cleandata_604/news/" + name
    abstract_path = Dir.res + "/cleandata_604/abstract/" + name

    lines = ftools.read_lines(text_path)

    absts = ftools.read_lines(abstract_path)
    res = []
    for i in range(len(absts)):
        max_v,max_index = 0,0
        for j in range(len(lines)):
            v =  tools.sim(absts[i],lines[j])
            if v >max_v:
                max_v = v
                max_index = j
        res.append(max_index)
    print(res)

    sens,tags =[],[]
    for line in lines:
        tmp0,tmp1 = tools.seperate_pog(line)
        sens.append(tmp0)
        tags.append(tmp1)
    gv = Graph_Vec()
    sensv,essayv = gv.vectorize(sens,tags)
    dist = tools.Dist()

    print(sensv[res[0]])
    print(sensv[res[1]])
    print(sensv[res[2]])
    print(essayv)

    print(dist.sim(sensv[res[0]],essayv))
    print(dist.sim(sensv[res[1]],essayv))
    print(dist.sim(sensv[res[2]],essayv))
    print("-----")
    for var in sensv:
        print(dist.sim(var,essayv))
    # print(sensv)
    # print(essayv)

