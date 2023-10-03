#!/usr/bin/env python
# coding: utf-8

# # Algorithm
import numpy as np
from numpy.lib.stride_tricks import as_strided
import itertools

import torch


def maxpool2d(A, kernel_size, stride=1, padding=0, return_indices=False):
    input = torch.from_numpy(A).unsqueeze(0).type(torch.float32)
    if return_indices:
        with torch.no_grad():
            output, indices = torch.nn.functional.max_pool2d(input, kernel_size, stride=stride, padding=padding,
                                                             dilation=1, ceil_mode=False, return_indices=True)
        output = output.squeeze().numpy()
        indices = indices.squeeze().numpy()
        return output, indices
    else:
        with torch.no_grad():
            output = torch.nn.functional.max_pool2d(input, kernel_size, stride=stride, padding=padding, dilation=1,
                                                    ceil_mode=False, return_indices=False)

        output = output.squeeze().numpy()
        return output


def max_pool2d(A, kernel_size, stride=1, padding=0, return_indices=False):
    '''
     2D MaxPooling

     Parameters:
         A: input 2D array
         kernel_size: int, the size of the window over which we take pool
         stride: int, the stride of the window
         padding: int, implicit zero paddings on both sides of the input
         return_indices: bool, return argmax
     '''
    # Padding
    A = np.pad(A, padding, mode='constant')

    # Window view of A
    output_shape = ((A.shape[0] - kernel_size) // stride + 1,
                    (A.shape[1] - kernel_size) // stride + 1)

    shape_w = (output_shape[0], output_shape[1], kernel_size, kernel_size)
    strides_w = (stride * A.strides[0], stride * A.strides[1], A.strides[0], A.strides[1])

    A_w = as_strided(A, shape_w, strides_w)
    A_w = A_w.reshape(output_shape[0], output_shape[1], kernel_size * kernel_size)

    A_max = A_w.max(axis=2)

    if return_indices:
        A_argmax = A_w.argmax(axis=2)

        R_w =  (A_argmax % kernel_size) - padding
        Q_w = (A_argmax // kernel_size - padding) * output_shape[0]
        L_w = np.arange(0, output_shape[0] * output_shape[1])
        L_w = L_w.reshape(output_shape[0], output_shape[1])
        return A_max, R_w + Q_w + L_w
    else:
        return A_max


def grouper(n, iterable, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return itertools.zip_longest(fillvalue=fillvalue, *args)

def gaussian2d(size=(256, 256), point=(0, 0), sigma=1):
    x_size, y_size = size
    x0, y0 = point
    x = np.linspace(0, x_size - 1, x_size)
    y = np.linspace(0, x_size - 1, y_size)

    x, y = np.meshgrid(x, y)

    z = (1 / (2 * np.pi * sigma ** 2) *
         np.exp(-((x - x0) ** 2 / (2 * sigma ** 2) +
                     (y - y0) ** 2 / (2 * sigma ** 2))))
    return z

def topo_gaussian_map(dgm, w, h):
    x_birth = dgm[:, 3]
    y_birth = dgm[:, 4]

    x_death = dgm[:, 5]
    y_death = dgm[:, 6]

    lifetime = dgm[:, 2]

    lpd = np.sqrt((x_birth - x_death) ** 2 + (y_birth - y_death) ** 2)

    out = np.zeros((w, h))
    for i in range(len(x_birth)):
        if lpd[i] > 1:
            out += lifetime[i] * gaussian2d(
                (h, w),
                (x_birth[i], y_birth[i]),
                np.sqrt(lpd[i])
            )

    return out


def persistence(img, return_mask=False):
    H, W = img.shape
    p, m = maxpool2d(img, kernel_size=3, stride=1, padding=1, return_indices=True)

    del p

    img = img.flatten()
    m_fo = m.flatten()

    m_f = m_fo[m_fo]
    while not np.array_equal(m_fo, m_f):
        m_fo = m_f
        m_f = m_fo[m_fo]

    del m_fo

    p1 = maxpool2d(m_f.reshape(H, W), kernel_size=3, stride=1, padding=1, return_indices=False)
    p2 = -maxpool2d((-m_f).reshape(H, W), kernel_size=3, stride=1, padding=1, return_indices=False)
    mask = p1 != p2
    mask = mask.flatten()

    del p1, p2

    bound_indices = np.nonzero(mask)[0]
    sort_indices = np.argsort(m_f[bound_indices])

    indices = bound_indices[sort_indices]
    del bound_indices, sort_indices

    pbirth, counts = np.unique(m_f[indices], return_counts=True)
    split_idxs = np.cumsum(counts, axis=0)[:-1]
    del counts

    split_list = np.split(img[indices], split_idxs)
    split_idxs = np.insert(split_idxs, 0, 0)

    to_process = [(pbirth[i], split_idxs[i], split_list[i]) for i in range(pbirth.size)]
    del pbirth, split_idxs, split_list

    dgm = []
    for elem in to_process:
        b = elem[0]
        d = indices[elem[1] + elem[2].argmax()]
        dgm.append([b, d])

    dgm = np.array(dgm)
    dgm = np.stack([
        img[dgm[:, 0]],
        img[dgm[:, 1]],
        dgm[:, 0] % W,
        dgm[:, 0] // W,
        dgm[:, 1] % W,
        dgm[:, 1] // W
        ], axis=1)

    if return_mask:
        return dgm, m_f.reshape(W, H)
    else:
        return dgm


# # Utils
from glob import glob
from pyspark import SparkConf, SparkContext
import numpy as np
from time import time

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
n1

n2 = experiment(data_path='data', n_imgs=None, ncores="2", message_maxSize="1024", maxResultSize="8196000000")
n2


n4 = experiment(data_path='data', n_imgs=None, ncores="4", message_maxSize="1024", maxResultSize="8196000000")
n4


n6 = experiment(data_path='data', n_imgs=None, ncores="6", message_maxSize="1024", maxResultSize="8196000000")
n6


n8 = experiment(data_path='data', n_imgs=None, ncores="8", message_maxSize="1024", maxResultSize="8196000000")
n8


n10 = experiment(data_path='data', n_imgs=None, ncores="10", message_maxSize="1024", maxResultSize="8196000000")
n10


n12 = experiment(data_path='data', n_imgs=None, ncores="12", message_maxSize="1024", maxResultSize="8196000000")
n12


n14 = experiment(data_path='data', n_imgs=None, ncores="14", message_maxSize="1024", maxResultSize="8196000000")
n14


n16 = experiment(data_path='data', n_imgs=None, ncores="16", message_maxSize="1024", maxResultSize="8196000000")
n16


n18 = experiment(data_path='data', n_imgs=None, ncores="18", message_maxSize="1024", maxResultSize="8196000000")
n18


n20 = experiment(data_path='data', n_imgs=None, ncores="20", message_maxSize="1024", maxResultSize="8196000000")
n20


n22 = experiment(data_path='data', n_imgs=None, ncores="22", message_maxSize="1024", maxResultSize="8196000000")
n22


n24 = experiment(data_path='data', n_imgs=None, ncores="24", message_maxSize="1024", maxResultSize="8196000000")
n24



import matplotlib.pyplot as plt
x = [2,4,8,12,16,20,24,28,32,36,40,44,48]
y = [n1,n2,n4,n6,n8,n10,n12,n14,n16,n18,n20,n22,n24]
plt.plot(x,y/60, 'go', x,y/60, 'g--')

plt.xlabel('N. cores')
plt.ylabel('Minutes')
plt.title('2 Workers')
#plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.axis([0.5, 30, 0, 20])
plt.xticks(np.arange(2, 33, step=2))
plt.grid(True)
plt.show()
