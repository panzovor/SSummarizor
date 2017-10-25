from src.tools import Tools as tools
from src.tools import  FileTools as ftools
import Dir

def rouge_1_simple(left, right):
    left_dict, right_dict = {}, {}
    word_count = 0
    for line in left:
        for w in line:
            if w not in left_dict:
                left_dict[w] = 0
            left_dict[w] += 1
            word_count += 1
            if word_count > 140:
                break
    # for key in left_dict.keys():
    #     print(key,left_dict[key])

    word_count = 0
    for line in right:
        for w in line:
            if w not in right_dict:
                right_dict[w] = 0
            right_dict[w] += 1
            word_count += 1
            if word_count > 140:
                break
    # for key in right_dict.keys():
    #     print(key,right_dict[key])

    count = 0
    all = 0
    # print(left_dict.keys())
    for key in left_dict.keys():
        all += left_dict[key]
        if key in right_dict.keys():
            count += min(left_dict[key], right_dict[key])

    return count / all


def rouge_2_simple(left, right):
    left_dict, right_dict = {}, {}
    all = 0
    for line in left:
        for i in range(len(line) - 1):
            w = line[i] + " " + line[i + 1]
            if w not in left_dict:
                left_dict[w] = 0
            left_dict[w] += 1
    for line in right:
        for i in range(len(line) - 1):
            w = line[i] + " " + line[i + 1]
            if w not in right_dict:
                right_dict[w] = 0
            right_dict[w] += 1
    count = 0
    for key in left_dict.keys():
        all += left_dict[key]
        if key in right_dict.keys():
            count += min(left_dict[key], right_dict[key])
    return count / all


def rouge_detail():
    fname = "trainning_3570.txt"
    content = ftools.read_lines(Dir.res+"/result/cleandata_small/Second Version/abstract_processed/" + fname)
    refence = ftools.read_lines(Dir.res+"/cleandata_small/ref_processed/" + fname)
    lines = [line.split(" ") for line in content]
    refen = [line.split(" ") for line in refence]
    # print(lines)
    # print(refen)
    rouge1 = rouge_1_simple(refen,lines)
    rouge2 = rouge_2_simple(refen,lines)
    print(rouge1,rouge2)

from src.baseline.FirstNSenetencesSummarizor import  FirstNSentencesSummarizor as FirstNSenetencesSummarizor


def work(name):
    arg1= name["n"]
    arg2 = name["m"]
    args = name["k"]
    return arg1+"_"+str(arg2.summarize("111111。2222222。333333"))+"_"+args

def test():
    name=  [{"n":"1","m":FirstNSenetencesSummarizor(),"k":"1"},
            {"n": "2", "m": FirstNSenetencesSummarizor(), "k": "2"}]
    p = multiprocessing.Pool(2)
    rsl = p.map(work,name)
    print(rsl)

import multiprocessing
if __name__ == "__main__":
    test()

