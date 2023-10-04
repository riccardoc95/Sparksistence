#!/usr/bin/env python
# coding: utf-8

from glob import glob
from pyspark import SparkConf, SparkContext
import numpy as np
from time import time

from ph0 import persistence


def get_files_path(string="data/*.npy"):
    files = glob(string)
    files = ['/lustre/home/rceccaroni/Notebooks/' + x for x in files]
    return files


def stop_sparkcontext(sc):
    try:
        sc.stop()
    except:
        pass


def start_sparkcontext(ncores="2", message_maxSize="1024", maxResultSize="4098000000"):
    conf = SparkConf().setAppName('SPH')\
                      .set("spark.rpc.message.maxSize", message_maxSize)\
                      .set("spark.executor.cores", ncores)\
                      .set("spark.driver.maxResultSize",maxResultSize)

    sc = SparkContext(conf=conf)
    return sc


def image_to_array(path):
    data = np.load(path)

    return data.astype(np.float32)


def experiment(data_path='data', n_imgs=None, ncores="2", message_maxSize="1024", maxResultSize="4098000000"):
    start = time()

    files = get_files_path(data_path+"/*.npy")
    if n_imgs is not None:
        files = files[0:n_imgs]
    sc = start_sparkcontext(ncores=ncores, message_maxSize=message_maxSize,maxResultSize=maxResultSize)
    rdd = sc.parallelize(files, len(files))
    rdd = rdd.map(lambda f: image_to_array(f))
    rdd = rdd.map(lambda x: persistence(x))
    result_list = rdd.collect()

    end = time()

    stop_sparkcontext(sc)

    return end - start


# # Experiments
sc.stop()

n1 = experiment(data_path='data', n_imgs=None, ncores="1", message_maxSize="1024", maxResultSize="8196000000")
print(n1)

n2 = experiment(data_path='data', n_imgs=None, ncores="2", message_maxSize="1024", maxResultSize="8196000000")
print(n2)


n4 = experiment(data_path='data', n_imgs=None, ncores="4", message_maxSize="1024", maxResultSize="8196000000")
print(n4)


n6 = experiment(data_path='data', n_imgs=None, ncores="6", message_maxSize="1024", maxResultSize="8196000000")
print(n6)


n8 = experiment(data_path='data', n_imgs=None, ncores="8", message_maxSize="1024", maxResultSize="8196000000")
print(n8)


n10 = experiment(data_path='data', n_imgs=None, ncores="10", message_maxSize="1024", maxResultSize="8196000000")
print(n10)


n12 = experiment(data_path='data', n_imgs=None, ncores="12", message_maxSize="1024", maxResultSize="8196000000")
print(n12)


n14 = experiment(data_path='data', n_imgs=None, ncores="14", message_maxSize="1024", maxResultSize="8196000000")
print(n14)


n16 = experiment(data_path='data', n_imgs=None, ncores="16", message_maxSize="1024", maxResultSize="8196000000")
print(n16)


n18 = experiment(data_path='data', n_imgs=None, ncores="18", message_maxSize="1024", maxResultSize="8196000000")
print(n18)


n20 = experiment(data_path='data', n_imgs=None, ncores="20", message_maxSize="1024", maxResultSize="8196000000")
print(n20)


n22 = experiment(data_path='data', n_imgs=None, ncores="22", message_maxSize="1024", maxResultSize="8196000000")
print(n22)


n24 = experiment(data_path='data', n_imgs=None, ncores="24", message_maxSize="1024", maxResultSize="8196000000")
print(n24)



import matplotlib.pyplot as plt
x = [2,4,8,12,16,20,24,28,32,36,40,44,48]
y = [n1,n2,n4,n6,n8,n10,n12,n14,n16,n18,n20,n22,n24]
plt.plot(x, y/60, 'go', x, y/60, 'g--')

plt.xlabel('N. cores')
plt.ylabel('Minutes')
plt.title('2 Workers')
#plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.axis([0.5, 30, 0, 20])
plt.xticks(np.arange(2, 33, step=2))
plt.grid(True)
plt.show()
