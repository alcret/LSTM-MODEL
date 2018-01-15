# coding:UTF-8
import urllib2
import urllib
import cookielib
import re

#=============最简单方式访问
# response = urllib2.urlopen("http://www.baidu.com")
# print(response.read())
#====================request方式访问
# request = urllib2.Request("http://www.baidu.com")
# response = urllib2.urlopen(request)
# print response.read()
#==============post方式传递参数
# values = {'usename':'msconfig','password':'zc123456'}
# data = urllib.urlencode(values)
# url = 'http://www.baidu.com'
#
# request = urllib2.Request(url,data)
# response = urllib2.urlopen(request)
# print response.read()
#========================debuglog模式
# httpHandle = urllib2.HTTPHandler(debuglevel=1)
# httpsHandle = urllib2.HTTPSHandler(debuglevel=1)
# opener = urllib2.build_opener(httpHandle,httpsHandle)
# urllib2.install_opener(opener)
# response = urllib2.urlopen("http://www.baidu.com")
#=========================获取cookie保存变量
# cookie = cookielib.CookieJar()
# handler = urllib2.HTTPCookieProcessor(cookie)
# opener = urllib2.build_opener(handler)
# response = opener.open('http://www.baidu.com')
# for item in cookie:
#     print 'Name='+item.name
#     print 'Value='+item.value
#=======================保存cookie到文件
# filename = 'cookie.txt'
# cookie = cookielib.MozillaCookieJar(filename)
# handler = urllib2.HTTPCookieProcessor(cookie)
# opener = urllib2.build_opener(handler)
# response = opener.open("http://www.baidu.com")
# cookie.save(ignore_discard=True,ignore_expires=True)
#=========================从文件中读取cookie
# cookie = cookielib.MozillaCookieJar()
# cookie.load('cookie.txt',ignore_discard=True,ignore_expires=True)
# req = urllib2.Request("http://www.baidu.com")
# opener = urllib2.build_opener(urllib2.HTTPCookieProcessor(cookie))
# response = opener.open(req)
# print response.read()
#=======================抓取糗事百科
page = 1
url = 'http://www.qiushibaike.com/hot/page/'+ str(page)
pattern = re.compile('.*?',re.S)
user_agent = "Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)"
handers = {'User-Agent':user_agent}
try:
    request = urllib2.Request(url,headers=handers)
    response = urllib2.urlopen(request)
    # print response.read()
    content = response.read().decode('utf-8')
    items = re.findall(pattern, content)
    print '==========='
    # print(content)
    for item in items:
        print item[0], item[1], item[2], item[3], item[4], 'hello'
except urllib2.URLError,e:
    if hasattr(e,"code"):
        print e.code
    if hasattr(e,"reason"):
        print e.reason



