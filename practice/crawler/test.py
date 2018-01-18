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

#==================
from  multiprocessing import Pool
import time

def f1(i):
    time.sleep(0.5)
    print(i)
    return i + 100

if __name__ == "__main__":
    pool = Pool(1000)
    for i in range(1,31):
        pool.apply(func=f1,args=(i,))