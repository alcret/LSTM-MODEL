# -- coding = utf-8
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

f = open('dataset_2.csv')
df = pd.read_csv(f)
data = df.iloc[:,2:10].values
#常量设置
rnn_unit = 10
input_size = 7
output_size = 1
lr = 0.0006
# print(data)

# s数据集设置

def get_train_data(batch_size = 60,time_step=20,train_begin=0,train_end = 5800):
    batch_index=[]  #数据批量索引
    data_train = data[train_begin:train_end]
    # print(data_train)
    normalized_train_data = (data_train - np.mean(data_train,axis=0))/np.std(data_train,axis=0)    #轴为0的标准化
    train_x,train_y=[],[]  #训练集准备
    for i in range(len(normalized_train_data)-time_step):
        if i % batch_size==0:
            batch_index.append(i)
        x = normalized_train_data[i:i+time_step,:7]
        y = normalized_train_data[i:i + time_step, 7, np.newaxis]
        # print(y)
        train_x.append(x.tolist())
        train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data) - time_step))
    # print(batch_index)
    return batch_index,train_x,train_y

# 测试数据
def get_test_data(time_step = 20,test_begin = 5800):
    data_test = data[test_begin:]
    mean = np.mean(data_test,axis=0)
    # print(mean)
    std = np.std(data_test, axis=0)
    normalized_test_data = (data_test-mean)/std #标准化
    #取样
    size = (len(normalized_test_data)+time_step-1)//time_step
    # print(size)
    test_x,test_y = [],[]
    for i in range(size-1):
        x = normalized_test_data[i*time_step:(i+1)*time_step,:7]
        y = normalized_test_data[i*time_step:(i+1)*time_step,7]
        test_x.append(x.tolist())
        test_y.extend(y)
    test_x.append((normalized_test_data[(i+1)* time_step:,:7]).tolist())
    test_y.append((normalized_test_data[i+1* time_step:,7]).tolist())
    return  mean,std,test_x,test_y
# 神经网络定义
#神经网络变量定义
weights = {
    'in':tf.Variable(tf.random_normal([input_size,rnn_unit])),
    'out':tf.Variable(tf.random_normal([rnn_unit,1]))
}

biases = {
    'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
    'out':tf.Variable(tf.constant(0.1,shape=[1,]))
}

#定义神经网络




if __name__ == '__main__':
    # get_train_data()
    get_test_data()