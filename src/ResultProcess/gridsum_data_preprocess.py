import re
import src.ResultProcess.CrawResult_process as processor
import Dir
import src.tools.FileTools as tools

def preprocess(filepath = Dir.resource+"/gridsum_data/result.csv",save_dir = Dir.resource+"/gridsum_data/"):
    content_path = save_dir+"news/"
    abstract_path = save_dir+"abstract/"
    regex = "。|\?|\!|？|！|\.\.\.\.\.\."
    with open(filepath,mode="r",encoding="utf-8") as file:
        lines = file.readlines()
        for i in range(lines.__len__()):
            print("----",lines[i])
            if i == 0:
                continue
            try:
                line = lines[i]
                tmp = line.split(",")
                abstract = tmp[1]
                content = tmp[-1]
                content_save_path = content_path+"training_"+str(i)+".txt"
                abstract_save_path = abstract_path+"training_"+str(i)+".txt"
                abstracts = re.sub(regex,"\n",abstract)
                contents = re.sub(regex,"\n",content)
                with open(abstract_save_path,mode="w",encoding="utf-8") as file_abstract:
                    file_abstract.write(abstracts)
                with open(content_save_path,mode="w",encoding="utf-8") as content_abstract:
                    content_abstract.write(contents)
                print("training",i)
            except:
                print(line)
                print(tmp[1])
                input()

preprocess()

