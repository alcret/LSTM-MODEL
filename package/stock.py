#coding=utf-8
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------数据处理
f = open('dataset_1.csv')
df = pd.read_csv(f)        #通过pandas读取csv文件
data = np.array(df['max'])  #将股票的最大值放到numpy数组中
data = data[::-1]           #将data进行倒序
#倒序以后的data形式
# [   99.98   104.39   109.13 ...,  3495.7   3503.65  3455.55]
# plt.figure()
# plt.rcParams['font.sans-serif']=['SimHei']   #用来正常显示中文标签
# plt.rcParams['axes.unicode_minus']=False     #若存在负号此选项可以用来正常显示负号
# plt.title('股票最大值')   #在2.x版本中需要加u，3.x中不需要
# plt.plot(data)
# plt.show()

# print(np.mean(np.max(data)-np.min(data)))

normalize_data = (data - np.mean(data)) / np.std(data)   #均值归一化操作（标准化）
#np.mean()计算沿指定轴的平均值       np.std() 计算标准差
# print(normalize_data)
normalize_data = normalize_data[:,np.newaxis]  #增加一个新的轴(相当于增加一个维度)
# print(normalize_data)

# 常数设置
time_step = 20  # 时间步
lr = 0.0006  #学习率
batch_size = 60 #每一批次训练的数量
input_size = 1  #输入维度
output_size = 1 #输出维度
rnn_unit = 10  # 隐藏层单元
module_file = "train\\ckpt"

train_x, train_y = [], []   #训练集定义

# 构造训练集
# normalize_data 长度为6111 ->遍历6090
# print(len(normalize_data))
for i in range(len(normalize_data) - time_step - 1):
    x = normalize_data[i:i + time_step]
    y = normalize_data[i + 1:i + 1 + time_step]
    train_x.append(x.tolist())
    train_y.append(y.tolist())
# print(normalize_data[0:20])
# train_y:[-0.5022433058986824], [-0.49971611355921974], [-0.4876742306581236], [-0.48609945035704044], [-0.4866275203981222], [-0.47499111984999953], [-0.46684375350188134],
# shape[-1,time_step,input_size]
# ----------------------------------------------定义神经网络静态变量
# 定义每批次的输入
X = tf.placeholder(tf.float32, [None, time_step, input_size])
# 格式[[[1],[1],[1]...20],[[1],[1],[1]...20],[[1],[1],[1]...20]......]
# print(tf.reshape(X,[-1,input_size]))
# 定义每批次的输入标签
Y = tf.placeholder(tf.float32, [None, time_step, output_size])
# 格式[[[1],[1],[1]...20],[[1],[1],[1]...20],[[1],[1],[1]...20]......]
# print(Y)

# 权重
weights = {
    'in': tf.Variable(tf.random_normal([input_size, rnn_unit]),name='weitht_in'),
    'out': tf.Variable(tf.random_normal([rnn_unit, 1]),name='weight_out')
}
# weigths = {'in': <tf.Variable 'Variable:0' shape=(1, 10) dtype=float32_ref>, 'out': <tf.Variable 'Variable_1:0' shape=(10, 1) dtype=float32_ref>}
# print(tf.random_normal([input_size,rnn_unit]))
# 偏执单元
biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[rnn_unit,]),name='biases_in'),
    'out': tf.Variable(tf.constant(0.1, shape=[1, ]),name='biases_out')
}
# biases = {'in': <tf.Variable 'Variable_2:0' shape=(10,) dtype=float32_ref>, 'out': <tf.Variable 'Variable_3:0' shape=(1,) dtype=float32_ref>}
# print(biases)
# print(weights['in'])

# ----------------------------------lstm图定义
def lstm(batch):
    w_in = weights['in']  # 选取输入的权重
    # <tf.Variable 'Variable:0' shape=(1, 10) dtype=float32_ref>
    b_in = biases['in']  # 选取输入的偏执量
    # <tf.Variable 'Variable_2:0' shape=(10,) dtype=float32_ref>
    input = tf.reshape(X, [-1, input_size])  # 将每批次输入的X进行降维 X[None,20,1] ---> [-1,1]
    # input = Tensor("Reshape:0", shape=(?, 1), dtype=float32)
    input_rnn = tf.matmul(input, w_in) + b_in  # 计算隐藏层    shape [-1,10]+[10,]
    # inpur_rnn = Tensor("add:0", shape=(?, 10), dtype=float32)
    input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit])  # 隐藏层变形 成三维张量(升维)
    cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit,1)
    # tf.nn.rnn_cell.BasicLSTMCell() API说明
    #tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True):
    # n_hidden表示神经元的个数，forget_bias就是LSTM们的忘记系数，如果等于1，就是不会忘记任何信息。如果等于0，就都忘记。
    # state_is_tuple默认就是True，官方建议用True，就是表示返回的状态用一个元祖表示
    init_state = cell.zero_state(batch, dtype=tf.float32)  # 将初始的值放到cell中
    output_rnn, final_state = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state, dtype=tf.float32)  # 创建由RNNCELL指定的循环神经网络cell
    #tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state, dtype=tf.float32)  API说明
    #dynamic_rnn(cell,inputs,sequence_length=None,initial_state=None,dtype=None,parallel_iterations=None,swap_memory=False,time_major=False,scope=None)
    #cell:RNNCell的一个实例
    #inputs:RNN输入， #inputs:[batch_size, max_time, size]如果time_major=Flase. [max_time, batch_size, size]如果time_major=True
    #sequence_length:（可选）矢量大小
    #initial_state:（可选）RNN初始状态
    #dtype:（可选）初始状态和预期输出的数据类型
    #parallel_iterations:（默认32）并行运行的迭代次数
    #swap_memory:交换前向推到中产生的Tensor
    #time_major:input和output张量的形状格式
    #scope:创建子图的VariableScope，默认rnn
    # 返回：（outputs, states）:output，[batch_size, max_time, num_units]如果time_major=False。 [max_time,batch_size,num_units]如果time_major=True。states:[batch_size, 2*len(cells)]或[batch_size,s]
    # outputs输出的是最上面一层的输出，states保存的是最后一个时间输出的states
    output = tf.reshape(output_rnn, [-1, rnn_unit])  # 输出层变形为二维张量（降维）
    w_out = weights['out']  # 最后一层权重
    b_out = biases['out']  # 最后一层偏执量
    pred = tf.matmul(output, w_out) + b_out  # 输出
    return pred, final_state
#pred.shape[-1,1]

# lstm(batch_size)

# ----------------------------训练模型
def train_lstm():
    global batch_size  # 为函数外的变量赋值需要global关键字
    with tf.variable_scope("sec_lstm") as scope:    #上下文管理器，对于创建变量的操作，true时为共享变量  tf.AUTO_REUSE
        pred,_ = lstm(batch_size)
    loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1])))  # 代价函数,梯度下降法，类似欧几里德距离算法
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)    #此函数是Adam优化算法：是一个寻找全局最优点的优化算法，引入了二次方梯度校正。相比于基础SGD算法，1.不容易陷于局部优点。2.速度更快
    #minimize(loss,global_step=None,var_list=None,gate_gradients=GATE_OP,aggregation_method=None,colocate_gradients_with_ops=False,name=None,grad_loss=None)
    #loss:需要最小化的值
    # global_step：可选，Variable在更新变量后以1递增
    #var_list ：可选列表，要更新的对象元祖最小化loss
    #gate_gradients：选择梯度计算
    #aggregation_method：指定用于组合渐变项的方法
    #colocate_gtadients_with_ops：若为true，尝试使用相应的操作来对梯度进行合并
    #name：返回操作的名称（可选）
    #grad_loss可选，保持计算为梯度为loss
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())   #初始化
        for i in range(1):  # 迭代一百次训练
            step = 0
            start = 0
            end = start + batch_size
            while (end < len(train_x)):
                _,loss_ = sess.run([train_op, loss], feed_dict={X: train_x[start:end], Y: train_y[start:end]})
                start += batch_size
                end = start + batch_size
                if step % 10 == 0:
                    print("训练次数:", i, "脚步", step, "损失函数:", loss_)
                    print("保存", saver.save(sess, module_file))
                step += 1
        writer = tf.summary.FileWriter(
            "C:/Users/wilbert/PycharmProjects/LSTM-MODEL/for_simple_test/test/path/to/log",
            tf.get_default_graph())
        print("训练结束")


train_lstm()
# ----------------------------------------------------------------------------------------------------------------------
# 预测模型
def prediction():
    with tf.variable_scope("sec_lstm", reuse=True):
        pred,_ = lstm(1)  #预测时只输入[1,time_step,input_size]的测试数据
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        # module_file = tf.train.latest_checkpoint(module_file)
        saver.restore(sess,module_file)

        # 训练集最后一行测试
        prev_seq = train_x[-1]
        predict = []
        for i in range(100):
            next_seq = sess.run(pred, feed_dict={X: [prev_seq]})
            predict.append(next_seq[-1])
            prev_seq = np.vstack((prev_seq[1:], next_seq[-1]))  #每次得到最后一个时间步的预测结果，与之前的数据加在一起，形成新的测试样本

        plt.figure()
        plt.plot(list(range(len(normalize_data))), normalize_data, color='r')
        plt.plot(list(range(len(normalize_data), len(normalize_data) + len(predict))), predict, color='b')
        plt.show()

prediction()

# if __name__ == '__main__':
    # train_lstm()
    # prediction()
