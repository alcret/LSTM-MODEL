import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


f = open('C:\\Users\\wutang\\Downloads\\1616f224-29f8-42b9-967a-27dfea52e2c4.xls')
df = pd.read_excel('C:\\Users\\wutang\\Downloads\\1616f224-29f8-42b9-967a-27dfea52e2c4.xls')
# data = pd.read_excel(f)

# print(df)
data =np.array(df[0:208])
# print(data)
c = data.reshape([-1,1,1])
print(c)