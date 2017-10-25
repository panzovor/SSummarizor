import re

class FirstNSentencesSummarizor():

    def __init__(self):
        self.name = "FirstNSentencesSummarizor"

    def summarize(self,essay,num=3):
        sentences=[]
        if not isinstance(essay, list):
            regex = "！|？|。|；|\.\.\.\.\.\."
            new_sentences = re.split(regex, essay)
            # print(sentences.__len__())
            for sen in new_sentences:
                if sen.strip().__len__() > 3:
                    sentences.append(sen.strip())
        else:
            # print("wrong  ")
            sentences = essay
        if num < sentences.__len__():
            return sentences[:num]
        else:
            return sentences
