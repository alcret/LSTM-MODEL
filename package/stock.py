import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------数据处理
f = open('dataset_1.csv')
df = pd.read_csv(f)
data = np.array(df['max'])
data = data[::-1]

# plt.figure()
# plt.plot(data)
# plt.show()

normalize_data = (data - np.mean(data)) / np.std(data)
# print(normalize_data)
normalize_data = normalize_data[:, np.newaxis]
# print(normalize_data)

# 常数设置
time_step = 20  # 时间步
lr = 0.0025
batch_size = 60
input_size = 1
output_size = 1
rnn_unit = 10  # 隐藏层单元

train_x, train_y = [], []

# 构造训练集
for i in range(len(normalize_data) - time_step - 1):
    x = normalize_data[i:i + time_step]
    y = normalize_data[i + 1:i + 1 + time_step]
    train_x.append(x.tolist())
    train_y.append(y.tolist())
# train_y:[-0.5022433058986824], [-0.49971611355921974], [-0.4876742306581236], [-0.48609945035704044], [-0.4866275203981222], [-0.47499111984999953], [-0.46684375350188134],

# ----------------------------------------------定义神经网络静态变量
# 定义每批次的输入
X = tf.placeholder(tf.float32, [None, time_step, input_size])
# print(X)
# 格式[[[1],[1],[1]...20],[[1],[1],[1]...20],[[1],[1],[1]...20]......]
# print(tf.reshape(X,[-1,input_size]))
# 定义每批次的输入标签
Y = tf.placeholder(tf.float32, [None, time_step, output_size])
# 格式[[[1],[1],[1]...20],[[1],[1],[1]...20],[[1],[1],[1]...20]......]
# print(Y)

# 权重
weights = {
    'in': tf.Variable(tf.random_normal([input_size, rnn_unit])),
    'out': tf.Variable(tf.random_normal([rnn_unit, 1]))
}

# weigths = {'in': <tf.Variable 'Variable:0' shape=(1, 10) dtype=float32_ref>, 'out': <tf.Variable 'Variable_1:0' shape=(10, 1) dtype=float32_ref>}
# print(tf.random_normal([input_size,rnn_unit]))
# 偏执单元
biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[rnn_unit, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[1, ]))
}


# biases = {'in': <tf.Variable 'Variable_2:0' shape=(10,) dtype=float32_ref>, 'out': <tf.Variable 'Variable_3:0' shape=(1,) dtype=float32_ref>}
# print(biases)
# print(weights['in'])
# ----------------------------------lstm定义
# print(X)

def lstm(batch):
    w_in = weights['in']  # 选取输入的权重
    # <tf.Variable 'Variable:0' shape=(1, 10) dtype=float32_ref>
    b_in = biases['in']  # 选取输入的偏执量
    # <tf.Variable 'Variable_2:0' shape=(10,) dtype=float32_ref>
    input = tf.reshape(X, [-1, input_size])  # 隐藏层输入
    # input = Tensor("Reshape:0", shape=(?, 1), dtype=float32)
    input_rnn = tf.matmul(input, w_in) + b_in  # 计算隐藏层
    # inpur_rnn = Tensor("add:0", shape=(?, 10), dtype=float32)
    input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit])  # 隐藏层变形 成三维张量
    cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    init_state = cell.zero_state(batch, dtype=tf.float32)  # 将初始的值放到cell中
    output_rnn, final_state = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state, dtype=tf.float32)  # 输出层计算
    output = tf.reshape(output_rnn, [-1, rnn_unit])  # 输出层变形为二维张量
    w_out = weights['out']  # 标记的权重
    b_out = biases['out']  # 标记的偏执量
    pred = tf.matmul(output, w_out) + b_out  # 预测的值
    # print(input)
    return pred, final_state


# lstm(batch_size)

# ----------------------------训练模型
def train_lstm():
    global batch_size  # 为函数外的变量赋值需要global关键字
    with tf.variable_scope("sec_lstm"):
        pred, _ = lstm(batch_size)
    loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1])))  # 代价函数
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(100):  # 迭代一百次训练
            step = 0
            start = 0
            end = start + batch_size
            while (end < len(train_x)):
                _, loss_ = sess.run([train_op, loss], feed_dict={X: train_x[start:end], Y: train_y[start:end]})
                start += batch_size
                end = start + batch_size

                if step % 10 == 0:
                    print("训练次数:", i, "脚步", step, " 损失函数:", loss_)
                    print("保存", saver.save(sess, 'test\\modle.ckpt'))
                step += 1
        print("训练结束")


train_lstm()


# ----------------------------------------------------------------------------------------------------------------------
# 预测模型
def prediction():
    with tf.variable_scope("sec_lstm", reuse=True):
        pred, _ = lstm(1)
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        saver.restore(sess, "test\\modle.ckpt")

        # 训练集最后一行测试
        prev_seq = train_x[-1]
        predict = []
        for i in range(100):
            next_seq = sess.run(pred, feed_dict={X: [prev_seq]})
            predict.append(next_seq[-1])
            prev_seq = np.vstack((prev_seq[1:], next_seq[-1]))

        plt.figure()
        plt.plot(list(range(len(normalize_data))), normalize_data, color='r')
        plt.plot(list(range(len(normalize_data), len(normalize_data) + len(predict))), predict, color='y')
        plt.show()


# prediction()

if __name__ == '__main__':
    prediction()

