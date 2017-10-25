from urllib import request as Request
import requests
# import httplib
import ssl
import re
import src.tools.FileTools as tools
import time
import Dir

### page_index =  11961
### 出现can't decode byte 0x8b in position 可以删除 Accept-Encoding','gzip, deflate
class Crawer():

    def __init__(self):
        self.all_text = 0
        self.fail_pages =set()
        self.canot_crawe_info =  Dir.res+"/craw_data/fail/"
        # self.canot_crawe_info =  Dir.res+"\data_extract\\aun_crawe_pages_info.txt"
        self.craw_result = Dir.res+"/craw_data/data/"

    def crawe_webpage(self,page_index):

        url = "http://weibo.cn/breakingnews?page="+str(page_index)
        # print(url)
        header = ["Host","weibo.cn","User-Agent","Mozilla/5.0 (Windows NT 10.0; WOW64; rv:54.0) Gecko/20100101 Firefox/54.0","Accept","text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8","Accept-Language","zh-CN,zh;q=0.8,en-US;q=0.5,en;q=0.3","Cookie","_T_WM=300ef73d21256863a6f19cd785d7aa3e; SUB=_2A250skSNDeRhGedG6loS-SbLzzuIHXVUXWzFrDV6PUJbkdANLRfYkW1Fz2qj3sW5l9pQ73kkGDI9Unx8UQ..; SUHB=0wn0DwlHw3OVwC; SCF=AssiNe-dpqXqVvraCuNcQAO1f1x_fkzCtCnmTHilssHVUW4mYeTL4IkFa-Ua3HUH6i3ZdBZ8tjpJeDTCCJ2SJgA.; SSOLoginState=1505113319","Connection","keep-alive","Upgrade-Insecure-Requests","1"]
        request = Request.Request(url)
        params = {}
        for i in range(header.__len__()-1):
            if i%2 == 0:
                request.add_header(header[i],header[i+1])
                params[header[i]] = header[i+1]
        response = Request.urlopen(request)
        html = response.read()
        # print(html)
        # print(html.decode('utf-8'))
        # input()
        html = html.decode( 'utf-8')
        # file = self.craw_result+"page"

        num = self.extract_info(html,page_index)
        self.all_text +=  num[0]
        print("page", page_index, num[0],num[1], self.all_text)
        return page_index, num[0],num[1],self.all_text

    def extract_info(self,string,index):
        regex = "【(.*?)】(.*?)<a href=\"(http.*?)[\u4e00-\u9fa5]*?\""
        infos = re.findall(regex,string)
        result = []
        un_crawe_info = []
        try_count = 3
        for info in infos:
            if info.__len__() == 3:
                all_url = info[-1]
                if "amp;" in all_url:
                    all_url = all_url.replace("amp;","")
                # print(all_url)
                content = self.url(all_url)
                try_time = 0
                if content!=None:
                    content = content.replace("\n", "")
                if (content == None or content.__len__()==0) and try_time<try_count:
                    while(try_time<try_count):
                        time.sleep(0.1)
                        content = self.url(all_url)
                        # print("try",try_time)
                        if content!=None:
                            content = content.replace("\n","")
                        if content != None and content.__len__() > 0:
                            break
                        try_time+=1
                # print(try_time,all_url,info[0])
                if content!=None:
                    output_string = content.replace("\n","")
                    # print("hahah",output_string.__len__())
                if  content != None and content.__len__()>0:
                    tmp = []
                    for i in range(2):
                        tmp.append(info[i])

                    content_try,content_try_time = 3,0
                    origin = self.get_origin_text(content)
                    origin = origin.replace("\n","")
                    # print("result",origin)
                    if origin.__len__()<100:
                        # print("s-fail",origin)
                        if index not in self.fail_pages:
                            un_tmp = []
                            for info_un in tmp:
                                un_tmp.append(info_un)
                                # un_crawe_info.append(info_un)
                            un_tmp.append(info[-1])
                            un_crawe_info.append(un_tmp)
                        continue
                    tmp.append(origin)
                    result.append(tmp)
        string = ""
        count = 0

        if un_crawe_info.__len__()>0 and index not in self.fail_pages:
            print("fail",un_crawe_info.__len__())
            fail_info = ""
            for tmp_fail in un_crawe_info:
                fail_info+= str(tmp_fail)+"\n"
            tools.write(self.canot_crawe_info+"page"+str(index), fail_info, mode="w")
            self.fail_pages.add(index)

        for info in result:
            count+=1
            # print("text",count)
            # print(str(info))
            string+= str(info)+"\n"
        if string.__len__()>0:
            tools.write(self.craw_result+"page"+str(index),string)
        return count,infos.__len__()

    def url(self,url_string):
        try:
            # print(url_string)
            code = requests.get(url_string).status_code
        except:
            return None
        if code  == 200 :
            return requests.get(url_string).text
        if "amp;" in url_string:
            url_string = url_string.replace("amp;","")
        url = requests.get(url_string).url
        try:
            response = Request.urlopen(url)
        except:
            # print("cannot tansfer")
            return None
        html = response.read()
        try:
            # print("url",url)
            html = html.decode("utf-8")
            html = html.replace("\n","")
            # print(tmp)
        except:
            return None
        return html

    def get_origin_text(self,content):
        # regex_rest = "下页</a> <a href=\"(.*?)\">余下全文"
        # all_url = re.findall(regex_rest,content)
        # if all_url.__len__()>0:
        #     content = self.url(all_url[0])
        #     if content == None:
        #         print("first fail")
        #         return None
        regex ="class=\"art_t\"> *(.*?)<"
        result = re.findall(regex,content)
        if result.__len__()>0:
            string = ""
            for res in result:
                string+= res.strip()+"\n"
            return string
        else:
            # print(content)
            # print("second fail")
            return self.judge_content(content)


    def judge_content(self,content):
        sentences = content.split("\n")
        result = []
        missed_count= 0
        for sentence in sentences:
            detail_sentence = sentence.split("<")
            for details in detail_sentence:
                if details.__len__()>30:
                    count = 0
                    for i in range(details.__len__()):
                        if details[i] >= '\u4e00' and details[i]<= '\u9fa5':
                            count+=1
                    # print(count,details.__len__())
                    # print(details)
                    if count >20:
                        result.append(details)
                    else:
                        if missed_count >4:
                            break
                        missed_count+=1
        # print(result)
        string = ""
        for res in result:
            string += res.strip()
        string = string.replace("br/>","")
        # print(string)
        return string
    def writeIntofile(self,file,content):
        file_object = open(file, 'w')
        file_object.write(content)
        file_object.close()





    # ult_s= "http://news.sina.cn/?sa=t124d2300076v71&wm=3049_a111&from=qudao&vt=1"
    # # ult_s = "http://news.sina.cn/?sa=t124d2303738v71&wm=3049_a111&from=qudao&vt=1"
    # con = url(ult_s)
    # resul = get_origin_text(con)
    # print(resul)

    # urlss = "http://ent.sina.cn/?sa=d1556360t64v33&amp;sid=105469&amp;vt=1&amp;wm=3049_a111"
    # contentt = requests.get(urlss).content
    # print(contentt)
    # print(contentt.decode("utf-8"))
    # crawe_webpage(now)


if __name__ == "__main__":
    pages = 100
    now = 1
    crawer = Crawer()
    # crawer.crawe_webpage(10)
    i,try_count,try_times = now,3,0
    while i < pages:
        # try:
        print(i)
        page = None
        page = crawer.crawe_webpage(i)
        if page[1] <=2:
            try_times+=1
            if try_times>try_count:
                i+=1
                try_times = 0
            else:
                crawer.all_text -= page[1]
        else:
            i+=1
            try_times = 0
        # except Exception as e:
        #     try_times+=1
        #     # print(e.with_traceback())
        #     print(str(e))
        #     if page!=None:
        #         crawer.all_text -= page[1]

