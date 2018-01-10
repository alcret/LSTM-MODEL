#coding=utf-8
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#读取数据
f = open("dataset_2.csv")
df = pd.read_csv(f)
# print(df)
data = df.iloc[:,2:10].values
# print(data)

#基础数据设置
input_size = 7
output_size = 1
rnn_unit = 10
lr = 0.0006


#做训练集
def get_train_data(batch_size=60,time_step=20,train_begin=0,train_end=5800):
    batch_index = []
    data_train= data[train_begin:train_end]
    normalized_train_data = (data_train - np.mean(data_train,axis=0)) / np.std(data_train,axis=0)   #正规划处理
    train_x,train_y=[],[]
    for i in range(len(normalized_train_data) - time_step):
        if i % batch_size == 0:
            batch_index.append(i)
        x = normalized_train_data[i:i+time_step,:7]
        y = normalized_train_data[i:i+time_step,7,np.newaxis]
        train_x.append(x.tolist())
        train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data) - time_step))
    return batch_index,train_x,train_y

#测试集的选取
def get_test_data(time_step=20,test_begin=5800):
    data_test = data[test_begin:]
    mean = np.mean(data_test,axis=0)
    std = np.std(data_test,axis=0)
    normalized_test_data = (data_test - mean) / std
    size = (len(normalized_test_data)+time_step-1) //time_step
    test_x,test_y= [],[]
    for i in range(size -1):
        x = normalized_test_data[i*time_step:(i+1)*time_step,:7]
        y = normalized_test_data[i*time_step:(i+1)*time_step,7]
        test_x.append(x.tolist())
        test_y.extend(y)
    test_x.append((normalized_test_data[(i+1)*time_step:,:7]).tolist())
    test_y.append((normalized_test_data[(i+1)*time_step:,7]).tolist())
    return mean,std,test_x,test_y



#神经网络的定义
weight = {
    'in':tf.Variable(tf.random_normal([input_size,rnn_unit])),
    'out':tf.Variable(tf.random_normal([rnn_unit,output_size]))
}

biases={ 'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
         'out':tf.Variable(tf.constant(0.1,shape=[1,]))
         }
#===========lstm定义
def lstm(X):
    batch_size = tf.shape(X)[0]
    time_step = tf.shape(X)[1]
    w_in = weight['in']
    b_in = biases['in']
    input = tf.reshape(X,[-1,input_size])
    input_rnn = tf.matmul(input,w_in)+b_in
    input_rnn = tf.reshape(input_rnn,[-1,time_step,rnn_unit])
    cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    init_state = cell.zero_state(batch_size,dtype=tf.float32)
    output_rnn,final_states = tf.nn.dynamic_rnn(cell,input_rnn,initial_state=init_state,dtype=tf.float32)









if __name__ == '__main__':
    get_train_data()