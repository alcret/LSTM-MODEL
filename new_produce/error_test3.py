# coding:utf-8
import numpy as np
import pandas as pd
import tensorflow as tf
import datetime
import MySQLForML as my
import pymysql


class DB:
    try:
        datas = my.MySQLForML.DBCreate()
        # print(datas)
    except Exception as e:
        print(e)

    def initials(df):
        # 全局变量定义
        global time_step, biases, Y, X, train_x, train_y, normalize_data, data,lr,input_size,output_size,batch_size,rnn_unit,module_file,weights

        data = np.array(df['dl_errorfirerate'])
        data = data[::-1]
        normalize_data = (data - np.mean(data)) / np.std(data)
        normalize_data = normalize_data[:, np.newaxis]
        # 常数设置
        time_step = 20  # lstm展开的步数，也就是输入词的个数
        lr = 0.00025
        input_size = 1
        output_size = 1
        batch_size = 20  # 批量大小
        rnn_unit = 10
        module_file = "train\\model.ckpt"

        train_x, train_y = [], []

        # print(len(normalize_data))
        for i in range(len(normalize_data) - time_step - 1):
            x = normalize_data[i:i + time_step]
            y = normalize_data[i + 1:i + 1 + time_step]
            train_x.append(x.tolist())
            train_y.append(y.tolist())

        X = tf.placeholder(tf.float32, [None, time_step, input_size], name='X')
        Y = tf.placeholder(tf.float32, [None, time_step, output_size], name='Y')

        with tf.name_scope("weightAndbiases"):
            weights = {
                'in': tf.Variable(tf.random_normal([input_size, rnn_unit]), name='weights_in'),
                'out': tf.Variable(tf.random_normal([rnn_unit, 1]), name='weights_out')
            }
            biases = {
                'in': tf.Variable(tf.constant(0.1, shape=[rnn_unit, ]), name='biases_in'),
                'out': tf.Variable(tf.constant(0.1, shape=[1, ]), name='biases_out')
            }

class LSTM:
    def lstm(batch):
        w_in = weights['in']  # 选取输入的权重
        b_in = biases['in']  # 选取输入的偏执量
        input = tf.reshape(X, [-1, input_size], name='input')  # 将每批次输入的X进行降维 X[None,20,1] ---> [-1,1]
        input_rnn = tf.matmul(input, w_in) + b_in  # 计算隐藏层    shape [-1,10]+[10,]
        input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit], name='input_rnn')  # 隐藏层变形 成三维张量(升维)
        cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit, 1)
        init_state = cell.zero_state(batch, dtype=tf.float32)  # 将初始的值放到cell中
        print("==================================")
        output_rnn, final_state = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state,
                                                    dtype=tf.float32)  # 创建由RNNCELL指定的循环神经网络cell
        output = tf.reshape(output_rnn, [-1, rnn_unit], name='output')  # 输出层变形为二维张量（降维）
        w_out = weights['out']  # 最后一层权重
        b_out = biases['out']  # 最后一层偏执量
        pred = tf.matmul(output, w_out, name='pred') + b_out  # 输出
        return pred, final_state

class TRAIN_LSTM:
    def train_lstm(names):
        lstm=LSTM().lstm()
        # tf.reset_default_graph()
        global batch_size  # 为函数外的变量赋值需要global关键字
        with tf.variable_scope(str(names)) as scope:
            pred, _ = lstm(batch_size)
        loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1])), name='loss')  # 代价函数,梯度下降法，类似欧几里德距离算法
        train_op = tf.train.AdamOptimizer(lr, name='train_op').minimize(
            loss)  # 此函数是Adam优化算法：是一个寻找全局最优点的优化算法，引入了二次方梯度校正。相比于基础SGD算法，1.不容易陷于局部优点。2.速度更快
        saver = tf.train.Saver(tf.global_variables(), name='train_saver')
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())  # 初始化
            for i in range(10000):  # 迭代一百次训练
                step = 0
                start = 0
                end = start + batch_size
                while (end < len(train_x)):
                    _, loss_ = sess.run([train_op, loss], feed_dict={X: train_x[start:end], Y: train_y[start:end]})
                    start += batch_size
                    end = start + batch_size
                    if step % 2 == 0:
                        print("训练次数:", i, "脚步", step, "损失函数:", loss_)
                        print("保存", saver.save(sess, module_file))
                    step += 1
            writer = tf.summary.FileWriter(
                "C:/Users/wilbert/PycharmProjects/LSTM-MODEL/for_simple_test/test/path/to/log",
                tf.get_default_graph())
            print("训练结束")

        writer.close()



class PREDICTIONS:
    def prediction(names):
        lstm = LSTM().lstm()
        with tf.variable_scope(str(names), reuse=True):
            pred, _ = lstm(1)  # 预测时只输入[1,time_step,input_size]的测试数据
        saver = tf.train.Saver(tf.global_variables(), name='psaver')
        with tf.Session() as sess:
            # module_file = tf.train.latest_checkpoint(module_file)
            saver.restore(sess, module_file)

            # 训练集最后一行测试
            prev_seq = train_x[-1]
            predict = []
            predicts = []
            for i in range(1):
                next_seq = sess.run(pred, feed_dict={X: [prev_seq]})
                predict.append(next_seq[-1])
                prev_seq = np.vstack((prev_seq[1:], next_seq[-1]))  # 每次得到最后一个时间步的预测结果，与之前的数据加在一起，形成新的测试样本

            # plt.figure()
            # plt.plot(list(range(len(normalize_data))), normalize_data, color='r')
            # plt.plot(list(range(len(normalize_data), len(normalize_data) + len(predict))), predict, color='b')
            # plt.show()

            for x in predict:
                predicts.append(x * np.std(data) + np.mean(data))
            print(predicts[0])
            save = pd.DataFrame(predicts)
            save.to_csv("test.csv")

            # print(save)
            return predicts

class SAVE:
    def saves(i, predicts):
        tomorrow = datetime.datetime.today() + datetime.timedelta(days=1)  # 获取明日的日期
        tomorrow = tomorrow.strftime("%Y-%m-%d")
        # print(tomorrow)
        try:
            DB = pymysql.connect("172.16.1.159", "hadoop", "hadoop", "dl_iot_bd_tianjin", charset='utf8')
            cursor = DB.cursor()
            effect = cursor.executemany(
                "insert into bdf_ml_warningschedule(dl_orgid,dl_orgname,dl_errorfirerate,dl_arisetime) values(%s,%s,%s,%s)",
                [(int(DB.datas[i]['dl_orgid'][0]), str(DB.datas[i]['dl_orgname'][0]), float(predicts[0]), tomorrow)])
            DB.commit()
            DB.close()
        except Exception:
            print('数据库读取失败')


def rnn(i, datas):
    one = DB()
    print(i)
    try:
        DB().initials(datas)
        print(datas)
    except Exception as e :
        print(e)
    # TRAIN_LSTM().train_lstm(i)
    # a = PREDICTIONS().prediction(i)
    # SAVE().saves(i,a)


if __name__ == '__main__':
        # print(datas[0])
        # rnn(0,datas[0])
    try:
        for i in range(len(DB().datas)):
            rnn(i, DB().datas[i])
            # print(DB().datas[i])
            # print(i)
    except Exception as e:
        print(e)
