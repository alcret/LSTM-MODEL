import json
import re

import scrapy
from urllib import parse

class ZhihuSpider(scrapy.Spider):
    name = "zhihu"
    allowed_domains = ["www.zhihu.com"]
    start_urls = ['https://www.zhihu.com/']
    headers = {
        'User-Agent':"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36",

    }

    def start_requests(self):
        '''
        重写start_requests，请求登录页面
        :return:
        '''
        return [scrapy.Request('https://www.zhihu.com/#signin',headers=self.headers,callback=self.login)]


    def login(self,response):
        '''
        先通过正则获取xsrf值，然后通过scrapy.Request请求验证页面获取验证码
        :param response:
        :return:
        '''
        response_text = response.text
        match_obj = re.match('.*name="_xsrf" value="(.*?)"',response_text,re.DOTALL)
        print(match_obj.group(1))
        xsrf=''
        if match_obj:
            xsrf = match_obj.group(1)
        if xsrf:
            post_data = {
                "_xsrf":xsrf,
                "phone_num":"13121210484",
                "password":"********",
                'captcha':'',
            }
            import time
            t = str(int(time.time() * 1000))
            captcha_url = "https://www.zhihu.com/captcha.gif?r={0}&type=login".format(t)
            #这里利用meta讲post_data传递到后面的response中
            yield scrapy.Request(captcha_url,headers=self.headers,meta={"post_data":post_data} ,callback=self.login_after_captcha)

    def login_after_captcha(self,response):
        '''
        将验证码写入到文件中，然后登录
        :param response:
        :return:
        '''
        with open("captcha.jpg",'wb') as f:
            f.write(response.body)
        try:
            from PIL import Image
            im = Image.open("captcha.jpg")
            im.show()

        except:
            pass
        #提示用户输入验证码
        captcha = input("请输入验证码>:").strip()
        #从response中的meta中获取post_data并赋值验证码信息
        post_data = response.meta.get("post_data")
        post_data["captcha"] = captcha
        post_url = "https://www.zhihu.com/login/phone_num"
        # 这里是通过scrapy.FormRequest提交form表单
        return [scrapy.FormRequest(
            url=post_url,
            formdata=post_data,
            headers=self.headers,
            callback=self.check_login,
        )]

    def check_login(self,response):
        '''
        验证服务器的返回数据判断是否成功,我们使用scrapy会自动携带我们登录后的cookie
        :param response:
        :return:
        '''
        text_json = json.loads(response.text)
        print(text_json)
        for url in self.start_urls:
            yield self.make_requests_from_url(url,dont_filter=True,header=self.headers)





if __name__ == '__main__':
    x = ZhihuSpider()
    x.start_requests()