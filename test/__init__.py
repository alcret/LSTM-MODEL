import numpy as np
import  tensorflow as tf

a = np.array([1,2,3])[np.newaxis,:]
# print(a.shape,'\n' ,a)

b = np.array([[1,2,3],[1,2,3],[1,2,3]])
c = np.array([1,2,3])
# print(b.shape)
# print(b)
# print(c.shape)
# print(c)
# print(b+c)

# print(b.sum(axis=1))
# print('train/'+'modle.ckpt')
module_file = tf.train.latest_checkpoint("\wutang\Desktop\ML_WORK\LSTM-MODEL\package\train\\","train")
print(module_file)