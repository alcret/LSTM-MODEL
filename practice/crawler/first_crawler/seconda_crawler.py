import urllib
import urllib.request

data = {}
data['word']='zc'
url_value = urllib.parse.urlencode(data)
url = 'http://www.baidu.com/s?'
full_url = url+url_value

data = urllib.request.urlopen(full_url).read()
data = data.decode('utf-8')
print(data)