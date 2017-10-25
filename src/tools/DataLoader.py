from src.tools import FileTools as tools
## 加载数据
## 将目录下的所有文章读入
## 输入：数据所在目录
## 输出：所有文章,类型为list:[文章1，文章2]
def load_files(self ,filedir):
    filenames ,data = []
    tools.get_filelist(filedir ,filenames)
    for file in filenames:
        data.append(tools.read(file))
    return data