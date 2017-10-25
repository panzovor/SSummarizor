
class words_bag_vector():

    def __init__(self):
        self.name = "words bag vector"
        self.words_bag =[]

    def vectorize(self,sens_words,sens_tag = None):
        self.words_bag.clear()
        for var in sens_words:
            for w in var:
                if w not in self.words_bag:
                    self.words_bag.append(w)
        sens_vector,essay_vector = [],[1]* len(self.words_bag)
        for sen in sens_words:
            tmp =[0] * len(self.words_bag)
            for i in range(len(self.words_bag)):
                if self.words_bag[i] in sen:
                    tmp[i] =1
            sens_vector.append(tmp)
        return sens_vector,essay_vector
