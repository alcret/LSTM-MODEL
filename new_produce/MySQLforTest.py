# coding:utf-8
import pymysql as mysql
import pandas as pd

class MySQL_LSTM:
    def setSQL(self,sql):
        self.sql = sql

    def read_Mysql(self):

