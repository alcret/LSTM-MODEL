# coding:utf-8
import pymysql
import pandas as pd


class MySQLForML:
    def init(self):
        self.sql1 = "select distinct(dl_orgid) from bgf_groupbywarningschedule"  # 查询出所有orgid

    def DBCreate():
        sql1 = "select distinct(dl_orgid) from bgf_groupbywarningschedule"  # 查询出所有orgid

        data = []
        print('读取中......')
        try:
            DB = pymysql.connect("172.16.1.159","hadoop","hadoop","dl_iot_bd_tianjin",charset='utf8')
            df = pd.read_sql(sql1,con=DB)

            for i in df['dl_orgid']:
                pd.set_option('precision',18)
                sql2 = "select dl_orgid,dl_orgname,dl_arisetime,dl_errorfirerate from bgf_groupbywarningschedule where dl_orgid="+str(i)
                dat = pd.read_sql(sql2,con=DB)
                NONE_dl_orgid = (dat['dl_errorfirerate'].isnull()) | (
                    dat['dl_errorfirerate'].apply(lambda x: str(x).isspace()))
                dat = dat[~NONE_dl_orgid]
                data.append(dat)
            print('读取结束')
            DB.close()
        except Exception as e:
            print(e)
            print("读取失败")
        return data


