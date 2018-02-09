# _*_ coding: utf-8 _*_

import scrapy
import urllib.request
from scrapy.http import Request,FormRequest

class LoginspdSpider(scrapy.Spider):
    name = "loginspd"
    allowed_domains = ["douban.com"]
    header = {"User-Agent","Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3278.0 Safari/537.36"}
    def start_requests(self):
        return [Request("https://accounts.douban.com/login",meta={"cookiejar":1},callback=self.parse)]

    def parse(self, response):
        print()