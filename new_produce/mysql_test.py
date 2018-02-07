import pymysql
import numpy as np
import pandas as pd
from decimal import *


# cursor = db.cursor()

# sql = "select dl_ID from bdf_alarminfodetail_2017"
#
# try:
#     db = pymysql.connect("172.16.1.159", "hadoop", "hadoop", "dl_iot_bd_tianjin",charset='utf8')
#     cursor = db.cursor()
#     # effect = cursor.execute("insert into bdf_ml_warningschedule(dl_orgid,dl_orgname,dl_errorfirerate,dl_arisetime)"    #写入数据库
#     #                             " values(%s,%s,%s,%s)",[(("127"),("中文"),("0.123"),("2017-11-05"))])
#
#     db.commit()
#     print()
#     # print(df)
# except Exception as e:
#     print(e)
#
#     # print(df)
#     db.close()


sql1 = "select distinct(dl_orgid) from bgf_groupbywarningschedule"
def DBCreate():
    print('读取中。。。。。。。。。')
    try:
        # colums = []
        DB = pymysql.connect("172.16.1.159","hadoop","hadoop","dl_iot_bd_tianjin",charset='utf8')
        df = pd.read_sql(sql1,con=DB)
        # print(df)
        data = []
        for i in df['dl_orgid']:
            # print(i)
            pd.set_option('precision',18)
            sql2 = "select dl_orgid,dl_orgname,dl_arisetime,dl_errorfirerate from bdf_ml_warningschedule where dl_orgid="+str(i)
            # print(sql2)
            dat = pd.read_sql(sql2,con=DB)
            # dat.round(100)

            data.append(dat)
            # print(data)
            # print(data.shape)
        # 0.009009009009009009
        print('读取结束++++++++++++++')
        DB.close()
    except Exception:
        print("读取失败")

    # dat['dl_errorfirerate'].round(12)
    # print(dat)
    # print(len(dat))
    # print(type(dat))

    # for i in range(len(dat)):
    #     print(i)
    return data


if __name__ == '__main__':
    DBCreate()
