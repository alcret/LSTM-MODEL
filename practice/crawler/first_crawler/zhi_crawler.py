import urllib2
import urllib

request = urllib2.Request('https://www.zhihu.com/')
response = urllib2.urlopen(request)
print response.read()