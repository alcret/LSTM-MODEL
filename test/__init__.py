import numpy as np

a = np.array([1,2,3])[np.newaxis,:]
# print(a.shape,'\n' ,a)

b = np.array([[1,2,3],[1,2,3],[1,2,3]])
c = np.array([1,2,3])
print(b.shape)
print(b)
print(c.shape)
print(c)
print(b+c)