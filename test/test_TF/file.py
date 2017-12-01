import numpy as np
import pandas as pd


f = open('test.xlsx')
df = pd.read_excel(f)
data = np.array(df)
print(data)