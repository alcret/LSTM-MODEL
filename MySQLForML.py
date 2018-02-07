# coding:utf-8
import pymysql
import pandas as pd


class MySQLForML:
    def DBCreate(self):
        sql1 = "select distinct(dl_orgid) from bgf_groupbywarningschedule"  # 查询出所有orgid
        data = []
        print('读取中......')
        try:
            DB = pymysql.connect("172.16.1.159","hadoop","hadoop","dl_iot_bd_tianjin",charset='utf8')
            df = pd.read_sql(sql1,con=DB)

            for i in df['dl_orgid']:
                pd.set_option('precision',18)
                sql2 = "select dl_orgid,dl_orgname,dl_arisetime,dl_errorfirerate from bdf_ml_warningschedule where dl_orgid="+str(i)

                dat = pd.read_sql(sql2,con=DB)


                data.append(dat)

            print('读取结束')
            DB.close()
        except Exception:
            print("读取失败")
        return data


if __name__ == '__main__':
    a = MySQLForML().DBCreate()
    print(a)