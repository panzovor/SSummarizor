from urllib import request as Request
import urllib.request
import requests
import Dir
from newspaper import Article
from src.tools import FileTools as  tools
from urllib import parse as parse
import re


class Craw_Format():

    def __init__(self):

        self.page_nums = 13072
        self.seperator = "|__|"

        self.save_dir = Dir.res+"/url_data/"
        self.url_file = self.save_dir+"urls.txt"
        self.url_clean_file = self.save_dir+"url_clean.txt"

        self.url = "http://weibo.cn/breakingnews?page="
        self.header = ['Host', 'weibo.cn', 'User-Agent',
                  'Mozilla/5.0 (Windows NT 6.3; WOW64; rv:54.0) Gecko/20100101 Firefox/54.0', 'Accept',
                  'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8', 'Accept-Language',
                  'zh-CN,zh;q=0.8,en-US;q=0.5,en;q=0.3', 'Cookie',
                  'SCF=Ah6oK9ne4mmUNoYw4kUuNRmslSDJZqMC8SFA5i4tUHBOxdAcSzsIBEEOZfx3fQNj0BgpLdQSDXoBtnymKFxl8KA.; SUHB=0z1B6sSFzJ07wI; _T_WM=7fe561e14961c07e54388eb18a1b0902; SUB=_2A2502RrGDeRhGedG6loS-SbLzzuIHXVUJaaOrDV6PUJbkdANLVmtkW0WkE6llUm_KXMeRq22wEZ0nvVBRQ..; SSOLoginState=1507682966; H5:PWA:UID=1',
                  'DNT', '1', 'Connection', 'keep-alive', 'Upgrade-Insecure-Requests', '1']
        self.params = {}
        for i in range(self.header.__len__() - 1):
            if i % 2 == 0:
                self.params[self.header[i]] = self.header[i + 1]

        self.url_regex = "【(.*?)】(.*?)<a href=\"(http.*?)[\u4e00-\u9fa5]*?\""

    def url_unqoate(self,url):
        url = parse.unquote(url)
        url = url.replace("amp;", "")
        # print(url)
        return url

    def craw_urls(self):
        start = 372
        for i in range(start,self.page_nums):
            request = Request.Request(self.url+str(i))
            for key in self.params.keys():
                request.add_header(key,self.params[key])
            response = Request.urlopen(request)

            html = response.read()
            html = html.decode('utf-8')
            infos = re.findall(self.url_regex, html)
            save_content = ""
            for info in infos:
                new_url = self.url_unqoate(info[-1])
                new_infor = [info[0], info[1],info[-1], new_url]
                save_content += self.seperator.join(new_infor) + "\n"

            tools.check_build_file(self.url_file)
            tools.write(self.url_file, content=save_content, mode="a")
            print(i,len(infos))

if __name__ == "__main__":
    crawer = Craw_Format()
    crawer.craw_urls()
