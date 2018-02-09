from urllib import request, parse
import socket
import urllib.error
from bs4 import BeautifulSoup
import requests

# data = bytes(urllib.parse.urlencode({'word':'hello'}),encoding='utf8')
# print(data)
# response = urllib.request.urlopen('http://httpbin.org/post',data=data)
# print(response.read())
#=======================================异常
# try:
#     response = urllib.request.urlopen('http://httpbin.org/get',timeout=0.1)
# except urllib.error.URLError as e:
#     if isinstance(e.reason,socket.timeout):
#         print('time out')
#         # print(response.read())

#===================================Request方式使用，添加header
# request = urllib.request.Request('https://python.org')
# response = urllib.request.urlopen(request)
# print(response.read().decode('utf-8'))


# url = 'http://httpbin.org/post'
# headers = {
#     'User-Agent': 'Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)',
#     'Host': 'httpbin.org'
# }
#
# dict = {
#     'name':'zhaofan'
# }
# data = bytes(parse.urlencode(dict),encoding='utf8')
# req = request.Request(url=url,data=data,headers=headers,method='POST')
# response= request.urlopen(req)
# print(response.read().decode('utf-8'))


#=======================================================================github

Base_URL = "https://github.com/login"
Login_URL = "https://github.com/session"

def get_github_html(url):
    '''获取登陆页的html和cookie'''

    response = requests.get(url)
    first_cookie = response.cookies.get_dict()
    return response.text,first_cookie

def get_token(html):
    soup = BeautifulSoup(html,'lxml')
    res = soup.find("input",attrs={'name':"authenticity_token"})
    token = res["value"]
    return token


def github_login(url,token,cookie):
    data = {
        "commit": "Sign in",
        "utf8": "✓",
        "authenticity_token": token,
        "login": "alcret@163.com",
        "password": "z502016631"
    }
    response = requests.post(url,data=data,cookies=cookie)
    print(response.status_code)
    cookie = response.cookies.get_dict()
    return cookie


if __name__ == '__main__':
    html,cookie = get_github_html(Base_URL)
    token = get_token(html)
    cookie = github_login(Login_URL,token,cookie)
    response = requests.get("https://github.com/settings/repositories",cookies=cookie)
    print(response.text)