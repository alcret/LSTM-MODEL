#!/usr/bin/env python
# import queue
# import threading
#
#
# # message = Queue.Queue(10)
# message = queue.Queue(10)
#
#
# def producer(i):
#     while True:
#         message.put(i)
#
#
# def consumer(i):
#     while True:
#         msg = message.get()
#
#
# for i in range(12):
#     t = threading.Thread(target=producer, args=(i,))
#     t = threading.Thread(target=consumer, args=(i,))
#     print("p",i)
#     print("c", i)
#     t.start()

# for i in range(10):
#     t = threading.Thread(target=consumer, args=(i,))
#     print("c",i)
#     t.start()

# ==================
# from  multiprocessing import Pool
# import time
#
# def f1(i):
#     time.sleep(0.5)
#     print(i)
#     return i + 100
#
# if __name__ == "__main__":
#     pool = Pool(1000)
#     for i in range(1,31):
#         pool.apply(func=f1,args=(i,))
# ==============================================================


# def test(i):
#     if i < 0:
#         return -i
#
#
# a = -3
#
# try:
#     test(0)
#     print(test(0))
# except Exception as e:
#     print('hello')

# ====================================
from random import choice

x = choice(['Hello World!', [1, 2, 'e', '3', 4]])
# print(x.count('e'))
# print(x)

__metaclass__ = type  # 使用新式类，旧式类与新式类区别，新式类在模块或脚本开始的地方放置此赋值语句 3.0中旧式类不存在了（只在2.x中使用）


class Person:
    def setName(self, name):
        self.name = name

    def getName(self):
        return self.name

    def greet(self):
        print('Hello,world! I\'m %s' % self.name)


# foo = Person()
# bar = Person()
# foo.setName('Lu SK')
# bar.setName('A')
# foo.greet()


# 私有化

class Secretive:
    def _inaccessible(self):
        print("Bet you can not see me...")

    def accssible(self):
        print("the secret message is:")
        self._inaccessible()


s = Secretive()


# s.accssible()

# s._Secretice__inaccessible()


# class c:
#     print("class c being defined....")

# a = c
#      继承
class Filter:
    def init(self):
        self.blocked = []
    def filter(self,sequence):
        return [x for x in sequence if x not in self.blocked]

class SPAMFilter(Filter):
    def init(self):
        self.blocked = ['SPAM']

f = Filter()
f.init()
print(f.filter([1,2,3]))

s = SPAMFilter()
s.init()
b = s.filter(['SPAM','egg','SPAM'])
# print(b)
# print(issubclass(SPAMFilter,Filter))
# print(SPAMFilter.__bases__)

class Calculator:
    def calculate(self,expression):
        self.value = eval(expression)

class Talker:
    def talk(self):
        print("Hi,my value is",self.value )
class TalkingCalculator(Calculator,Talker):
    pass


tc = TalkingCalculator()
tc.calculate('1+2*3')
tc.talk()
# print(tc.talk())


if __name__ == '__main__':
    print(__name__)