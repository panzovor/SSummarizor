from urllib import request as Request
import urllib.request
import requests
import Dir
from newspaper import Article
from src.tools import FileTools as  tools
from urllib import parse as parse
import re

### page_index =  11961
### 出现can't decode byte 0x8b in position 可以删除 Accept-Encoding','gzip, deflate
class Crawer():

    def __init__(self):
        self.all_text = 0
        self.fail_pages =set()
        self.canot_crawe_info =  Dir.res+"/craw_data/fail/"
        # self.canot_crawe_info =  Dir.res+"\data_extract\\aun_crawe_pages_info.txt"
        self.craw_result = Dir.res+"/craw_data/data/"

    def clean(self,text):
        return text.strip().replace("\n","。").replace(",","，")

    def url_unqoate(self,url):
        url = parse.unquote(url)
        url = url.replace("amp;", "")
        # print(url)
        return url

    def craw_url(self,page_index,save_path):
        url = "http://weibo.cn/breakingnews?page=" + str(page_index)
        header = ['Host', 'weibo.cn', 'User-Agent',
                  'Mozilla/5.0 (Windows NT 6.3; WOW64; rv:54.0) Gecko/20100101 Firefox/54.0', 'Accept',
                  'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8', 'Accept-Language',
                  'zh-CN,zh;q=0.8,en-US;q=0.5,en;q=0.3', 'Cookie',
                  'SCF=Ah6oK9ne4mmUNoYw4kUuNRmslSDJZqMC8SFA5i4tUHBOxdAcSzsIBEEOZfx3fQNj0BgpLdQSDXoBtnymKFxl8KA.; SUHB=0z1B6sSFzJ07wI; _T_WM=7fe561e14961c07e54388eb18a1b0902; SUB=_2A2502RrGDeRhGedG6loS-SbLzzuIHXVUJaaOrDV6PUJbkdANLVmtkW0WkE6llUm_KXMeRq22wEZ0nvVBRQ..; SSOLoginState=1507682966',
                  'DNT', '1', 'Connection', 'keep-alive', 'Upgrade-Insecure-Requests', '1']
        request = Request.Request(url)
        params = {}
        for i in range(header.__len__() - 1):
            if i % 2 == 0:
                request.add_header(header[i], header[i + 1])
                params[header[i]] = header[i + 1]
        response = Request.urlopen(request)
        html = response.read()
        html = html.decode('utf-8')
        regex = "【(.*?)】(.*?)<a href=\"(http.*?)[\u4e00-\u9fa5]*?\""
        infos = re.findall(regex, html)
        save_content = ""
        for info in infos:
            new_url = self.url_unqoate(info[-1])
            reheader = requests.head(new_url).headers
            if "Location" in reheader:
                reurl = reheader["Location"]
            else:
                reurl = new_url
            if "pic" in reurl or "vedio" in reurl:
                continue
            new_infor = [info[0], info[1],reurl]

            save_content+= '\t'.join(new_infor)+"\n"

        tools.check_build_file(save_path)
        tools.write(save_path,content=save_content,mode="a")
        return len(infos)

    def crawe_webpage(self,page_index):

        url = "http://weibo.cn/breakingnews?page="+str(page_index)
        header = ['Host','weibo.cn','User-Agent','Mozilla/5.0 (Windows NT 6.3; WOW64; rv:54.0) Gecko/20100101 Firefox/54.0','Accept','text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8','Accept-Language','zh-CN,zh;q=0.8,en-US;q=0.5,en;q=0.3','Cookie','SCF=Ah6oK9ne4mmUNoYw4kUuNRmslSDJZqMC8SFA5i4tUHBOxdAcSzsIBEEOZfx3fQNj0BgpLdQSDXoBtnymKFxl8KA.; SUHB=0nlvviAH8qS7D9; SUB=_2A250WbVcDeRhGedG6loS-SbLzzuIHXVXpdsUrDV6PUJbktAKLU3ekW1d_dgMURgqrEu-lgeP-67j925GTg..; SUBP=0033WrSXqPxfM725Ws9jqgMF55529P9D9W5QbFqOSHXjH.sP8jAkPJVV5JpX5oz75NHD95Qp1h2Re0.RS0BNWs4DqcjZ-c2_dPHLdh.7Sntt; _T_WM=7fe561e14961c07e54388eb18a1b0902','DNT','1','Connection','keep-alive','Upgrade-Insecure-Requests','1']
        request = Request.Request(url)
        params = {}
        for i in range(header.__len__()-1):
            if i%2 == 0:
                request.add_header(header[i],header[i+1])
                params[header[i]] = header[i+1]
        response = Request.urlopen(request)
        html = response.read()
        html = html.decode( 'utf-8')
        regex = "【(.*?)】(.*?)<a href=\"(http.*?)[\u4e00-\u9fa5]*?\""
        infos = re.findall(regex, html)
        content = ""
        fail_content = ""
        count = 0
        fail_count = 0
        for info in infos:
            url = self.url_unqoate(info[-1])
            article = self.get_article(url)
            article = self.clean(article)
            # print(len(article))
            if len(article) > len(info[1]):
                count+=1
                content += info[0]+","+info[1]+","+article+"\n"
            else:
                fail_count+=1
                fail_content+= info[0]+","+info[1]+","+info[2]+"\n"
        print("succ",count,"fail",fail_count)
        self.writeIntofile(Dir.res+"/craw_data/data/page"+str(page_index),content)
        self.writeIntofile(Dir.res+"/craw_data/fail/page"+str(page_index),fail_content)
        return infos

    def get_article(self,url="https://weibo.cn/sinaurl?f=w&u=http://t.cn/RpoojkR&ep=FltrHC4h1,1618051664,FltrHC4h1,1618051664"):
        # print(url)
        try:
            a = Article(url, language='zh')  # Chinese

            a.download()
            a.parse()
            return a.text.replace("\n\n", "。").replace(",", "，")
            # print(a.text.replace("\n\n","。"))
        except:
            return ""


    def writeIntofile(self,file,content,mode = "w"):
        file_object = open(file, mode = mode)
        file_object.write(content)
        file_object.close()

def fill_all(path = Dir.res+"/craw_data/original/",save_path = Dir.res+"/craw_data/data/",fail_save_path = Dir.res+"/craw_data/fail/"):
    crawer = Crawer()
    files = tools.get_files(path)
    for name in files:
        content = tools.read_lines(path+name)
        fail_content = ""
        save_content = ""
        crawer.writeIntofile(save_path+name,"")
        crawer.writeIntofile(fail_save_path+name,"")
        succ_count,fail_count = 0,0
        for line in content:
            tmp = line.split(",")
            article = crawer.get_article(tmp[-1]).strip().replace("\n","")
            if len(article)>0:
                save_content += tmp[0]+","+tmp[1]+","+article+'\n'
                succ_count+=1
            else:
                fail_content += tmp[0]+","+tmp[1]+","+tmp[2]+'\n'
                fail_count+=1
                # fail_content.append(tmp)
        crawer.writeIntofile(save_path+name,save_content)
        crawer.writeIntofile(fail_save_path+name,fail_content)
        print(name,succ_count,fail_count)






if __name__ == "__main__":
    pages = 10000
    start = 4128
    crawer = Crawer()
    # for i in range(start,pages):
    #     print(i)
    #     crawer.crawe_webpage(i+1)


    ## sina vedio :weibo.com/tv/v/FobVN3dz1?fid=1034:001157b49c1ae78123786b9f4747ce06a
    ## other vedio:http://www.miaopai.com/show/HyMfZLZWLquHUgA57~rs460fI0F6ODKwcaFqHg__.htm
    ##http://news.sina.cn/?sa=t124d12328480v71&wm=3049_a111&from=qudao
    ##https://news.sina.cn/sh/2017-10-11/detail-ifymrcmn0144170.d.html?wm=3049_0015
    # article = crawer.get_article(url = "http://news.sina.cn/?sa=t124d12328480v71&wm=3049_a111&from=qudao")
    # print(article)

    # savepath = Dir.res+"/url_data/urls.txt"
    # for i in range(pages):
    #
    #     infos_length = crawer.craw_url(i,savepath)
    #     print(i,infos_length)


    articles = crawer.get_article("https://news.sina.cn/gn/2014-11-17/detail-icczmvum9948851.d.html?wm=3049_0015")
    print(articles)

    # crawer = Crawer()
    # res = crawer.get_article()
    # print(res)


    # i = 1
    # file = Dir.res+"/craw_data/original/page"
    # crawer.get_article()

    # fill_all()

    # while i <= pages:
    #     page = crawer.crawe_webpage(i)
    #     print(i,len(page))
    #     content = ""
    #     for info in page:
    #         # crawer.writeIntofile()
    #         content += info[0].replace(",","，")+","+info[1].replace(",","，")+","+info[2].replace(",","，")+"\n"
    #     crawer.writeIntofile(file+str(i),content,mode="w")
    #     i+=1