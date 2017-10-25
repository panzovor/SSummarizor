import jieba
import src.tools.FileTools as tools
import Dir
def preprocess(essay):
    if essay.strip().__len__() == 0:
        return None
    sentences = essay.split("！|？|。|；|\.\.\.\.\.\.")
    result =[]
    for sentence in sentences:
        if sentence.strip().__len__() >0:
            words = list(jieba.cut(sentence))
            result.append(' '.join(words))
    return result


def preprocess_file(file, savepath):
    content = tools.read(file)
    result = preprocess(content)
    tools.write_list(savepath,result)


def build_w2v_train_data():
    file_dir = Dir.res+"data/news.sentences/"
    save_path = Dir.res+"data/all.txt"
    filelist =[]
    content =[]
    tools.get_filelist(file_dir,filelist)
    for file in filelist:
        sentences  = tools.read_lines(file)
        content.extend(sentences)
    tools.write_list(save_path,content)
build_w2v_train_data()