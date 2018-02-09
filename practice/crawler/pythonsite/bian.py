from urllib import request, parse
import socket
import urllib.error
from bs4 import BeautifulSoup
import requests
import json



Base_URL = "http://www.coinbene.com/api/service/006-001"

def get_BiAn_html(url):
    response = urllib.request.urlopen(url)
    s = json.load(response)
    print(s.keys())
    for i in range(len(s["dayPrices"])):
        # print(s["dayPrices"][i]["englishName"])
        if  s["dayPrices"][i]["englishName"] == "SWTC":
            print("中文名称：",s["dayPrices"][i]["chineseName"])
            print("英文名称：",s["dayPrices"][i]["englishName"])
            print("最高价格：",s["dayPrices"][i]["highPrice"])
            print("增幅：",s["dayPrices"][i]["increase"])
            print("最近买入：",s["dayPrices"][i]["latestBuyPrice"])
            print("最低价格：",s["dayPrices"][i]["lowPrice"])
            print("现在价格：",s["dayPrices"][i]["nowPrice"])
            print("开始价格：",s["dayPrices"][i]["openPrice"])
            print("最近：",s["dayPrices"][i]["percent"])
            print("价格精度：",s["dayPrices"][i]["pricePrecision"])

    # print(s["dayPrices"][3])
    # print(st["chineseName"])
    # soup=BeautifulSoup(response,'lxml')
    # for name in soup.find_all('name'):
    #     print(name)
    # print(type(response.read()))
    # print(response.read())
#===========================================================
# response = urllib.request.urlopen('http://www.baidu.com')
# print(response.read().decode('utf8'))
#===============================================================requests

response = requests.get("http://www.baidu.com")
a  = response.content
print(a)

# if __name__ == '__main__':
        # get_BiAn_html(Base_URL)