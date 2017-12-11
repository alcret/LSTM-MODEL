# ReadMe

## 传统的神经网络

* 传统的神经网络无法对已经发生的数据进行记忆（记忆的数据作为下一批次的参量传入），所谓的时间序列推断后续事件

* RNN包含循环的网络，允许信息的持久化

  RNN特性：连接先前的信息到当前的任务上。

  缺点：当间隔不断增大时，RNN会丧失学习到连接如此远的信息的能力。

## 一、LSTM网络

一种RNN特殊的类型，可以学习长期依赖信息。

与RNN区别：标准的RNN中，重复的模块是一个非常简单的结构（例：重复模块包含单一层）。对比于LSTM是同样的结构，但是重复的模块是一个不同的结构（重复模块包含四个交互层）。

### 结构

门：去除或增加信息到细胞状态的能力，包含一个sigmoid神经网络层和一个pointwise乘法操作。Sigmoid层输出0~1的值，描述每个部分有多少量可以通过。LSTM拥有三个门（忘记门、输入门层、更新），来保护和控制细胞状态。

### 发展趋势

* 注意力结构的加入

## 二、共享变量

训练深度神经网络时，为了减少需要训练参数的个数、或多机多卡并行化训练大数据模型等情况时往往需要共享变量。

1. tf.Variable()与tf.get_Variable()作用与区别：

* tf.Variable()会自行检测命名冲突并自动处理。但tf.get_Variable()遇到重名的变量名，且变量名没有设置为共享变量时会报错。（拥有变量检查机制）

2. tf.name_scope(<scope_name>)与tf.variable_scope(<scope_name>)：

* tf.name_scope(<scope_name>):主要用于管理一个图里面的各种op，返回的是一个以scope_name命名的context manager。f.variable_scope(<scope_name>)：一般与tf.name_scope()配合使用，用于管理一个graph中变量的名字，避免变量之间的命名冲突，tf.variable_scope(<scope_name>)允许在一个variable_scope下面共享变量。需要注意的是：创建一个新的variable_scope时不需要把reuse属性设置未False，只需要在使用的时候设置为True就可以了。

------

### 三、CKPT文件

TensorFlow模型会保存在后缀为.ckpt的文件中，由于TensorFlow会将计算图上的结构和图上的参数取值分开保存，保存后的文件夹会出现三个文件，

1. checkpoint：保存了一个目录下所有的模型文件列表
2. model.ckpt.meta:文件保存了计算图结构，类似于圣经网络的 网络结构
3. model.ckpt：保存了程序的每个变量取值。