#coding=utf-8


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def residual(t,x,y):
    return y-(t[0]*x**2+t[1]*x+t[2])

def residual2(t,x,y):
    print(t[0],t[1])
    return y-(t[0]*np.sin(t[1]*x)+t[2])

def f(x):
    y=np.ones_like(x)
    i=x>0
    y[i] = np.power(x[i],x[i])
    i=x<0
    y[i] = np.power(-x[i],-x[i])
    return y



if __name__ == '__main__':

    a = np.arange(0,60,10).reshape((-1,1))+np.arange(6)
    # print(a)

    L = [1,2,3,4,5,6]
    # print(L)
    a = np.array(L)
    # print(a)
    # print(type(a),type(L))

    b = np.array([[1,2,3,4],[5,6,7,8]])
    # print(b)
    # b.reshape([1,-1])
    # print(b)
    # print(type(b))
    # print(b.dtype)

    np.set_printoptions(linewidth=100,suppress=True)
    a = np.arange(1,10,0.5)
    # print(a)

    line = np.linspace(1,10,10,endpoint=False)
    # print(line)
    c = np.logspace(0,10,11,endpoint=True,base=2)
    # print(c)
    s = 'abcdzzzz'
    g = np.fromstring(s, dtype=np.int8)
    # print(g)
    a = np.arange(0,60,10).reshape(-1,1)+np.arange(6)
    # print('a=',a)
    # b = a.reshape((-1,1))
    # print('b=',b)
    # c = np.arange(6)
    # print('c=',c)
    # print(b+c)
    # print(a)
    # print(a[[0, 1, 2], [2, 3, 4]])
    # print(a[4, [2, 3, 4]])
    # print(a[4:, [2, 3, 4]])

    a = np.arange(1, 7).reshape((2, 3))
    b = np.arange(11, 17).reshape((2, 3))
    c = np.arange(21, 27).reshape((2, 3))
    d = np.arange(31, 37).reshape((2, 3))
    print('a = \n', a)
    print('b = \n', b)
    print('c = \n', c)
    print('d = \n', d)

    s = np.stack((a,b,c,d),axis=1)
    print('s=',s)
