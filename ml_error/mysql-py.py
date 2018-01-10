import pymysql
import numpy as np
import pandas as pd


# cursor = db.cursor()

sql = "select dl_ID from bdf_alarminfodetail_2017"

try:
    db = pymysql.connect("172.16.1.159", "hadoop", "hadoop", "dl_iot_bd_tianjin",charset='utf8')
    cursor = db.cursor()
    # cursor.execute(sql)
    # results = cursor.fetchall()
    # for row in results:
    #     id = row[0]
    #     print(id)
    # df = pd.read_sql("select dl_ID, dl_arisetime,dl_errorfirerate from bdf_ml_warningschedule where dl_orgid=127", con=db,index_col='dl_ID')
    # df = pd.read_sql_table(table_name="bdf_ml_warningschedule",con=db)
    # effect = pd.read_sql("insert into bdf_ml_warningschedule(dl_orgid,dl_orgname,dl_errorfirerate,dl_arisetime) values('127','hello','0.123','2018-01-11')",con=db)
    # effect = cursor.execute( "insert into bdf_ml_warningschedule(dl_orgid,dl_orgname,dl_errorfirerate,dl_arisetime) values('127','hello','0.123','2018-01-11')")
    effect = cursor.execute("insert into bdf_ml_warningschedule(dl_orgid,dl_orgname,dl_errorfirerate,dl_arisetime)"  
                                " values(%s,%s,%s,%s)",[(("127"),("中文"),("0.123"),("2017-11-05"))])

    db.commit()
    print(effect)
    # print(df)
except Exception as e:
    print(e)

    # print(df)
    db.close()
