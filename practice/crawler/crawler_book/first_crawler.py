import urllib.request

#
# file = urllib.request.urlopen("http://www.baidu.com")
# data = file.read()
# dataline = file.readline()
# print(file.info(),sep='fdsa',end='dfsa',file=None,flush=False)
# # fhandle = open("C:/1.html","wb")
# # fhandle.write(data)
# # fhandle.close()


#================================scdn
# url = "http://blog.csdn.net/a2Ni5KFDaIO1E6/article/details/79070511"
#
# headers = ("User-Agent","Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3278.0 Safari/537.36")
# opener = urllib.request.build_opener()
# opener.addheaders = [headers]
# data = opener.open(url).read()
# print(data)
# fhandle = open("C:/csdn.html","wb")
# fhandle.write(data)
# fhandle.close()


# request = urllib.request.urlopen(url)
# print(request.read())




#===================================================baidu get请求
keyword = "hello"
url = "http://www.baidu.com/s?wd="+keyword
req = urllib.request.Request(url)
data = urllib.request.urlopen(req).read()
fhandle = open("C:/baidu.html","wb")
fhandle.write(data)
fhandle.close()