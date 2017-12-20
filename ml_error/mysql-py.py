import pymysql

db = pymysql.connect("172.16.1.159","hadoop","hadoop","dl_iot_bd_tianjin")
cursor = db.cursor()

sql = "select dl_ID from bdf_alarminfodetail_2017"

try:
    cursor.execute(sql)
    results = cursor.fetchall()
    for row in results:
        id = row[0]
        print(id)

except:
    print("error")

db.close()
