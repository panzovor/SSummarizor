import re
from src.tools import  Tools as tools

class FirstNSentencesSummarizor():

    def __init__(self):
        self.name = "FirstNSentencesSummarizor"
        self.info = self.name

    def summarize(self,essay,num=3,fname = None):
        sentences = tools.seperate_sentences(essay)
        return sentences[:num]
