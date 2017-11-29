import tensorflow as tf

a = tf.constant([[1,2,3,4,5,6],[1,2,3,4,5,6]],shape=[2,6],name='test')
# print(a)
b = tf.shape(a,name='b')
# print(b)
c = tf.reshape(b,[-1])
# print(c)


d = tf.constant([[[1,1,1,1],[2,2,2,2]],[[3,3,3,3],[4,4,4,4]],[[5,5,5,5],[6,6,6,6]],[[7,7,7,7],[8,8,8,8]]],name='t')
# print(d)
e = tf.reshape(d,[2,4,4])
# print(e)

f = tf.constant([[0,1,2,3,4,5,6,7,8,9]])
g = tf.constant([[0],[1],[2],[3],[4],[5]])
# print(g)

# shape=(10, 1) * shape=(1, 10)

h = tf.ones([1,10])