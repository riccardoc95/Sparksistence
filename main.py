# Import libraries
from pyspark.sql import SparkSession
import numpy as np
import pandas as pd
from astropy.io import fits


def neighbors(i, include_center=False):
    size = w.value * h.value
    neighbors = []
    if i - w.value >= 0:
        neighbors.append(i - w.value)  # north
    if i % w.value != 0:
        neighbors.append(i - 1)  # west

    if (i + 1) % w.value != 0:
        neighbors.append(i + 1)  # east

    if i + w.value < size:
        neighbors.append(i + w.value)  # south

    if ((i - w.value - 1) >= 0) and (i % w.value != 0):
        neighbors.append(i - w.value - 1)  # northwest

    if ((i - w.value + 1) >= 0) and ((i + 1) % w.value != 0):
        neighbors.append(i - w.value + 1)  # northeast

    if ((i + w.value - 1) < size) and (i % w.value != 0):
        neighbors.append(i + w.value - 1)  # southwest

    if ((i + w.value + 1) < size) and ((i + 1) % w.value != 0):
        neighbors.append(i + w.value + 1)  # southeast

    if include_center:
        neighbors.append(i)
    return neighbors


def fmap_argmax_neighbors(i, include_center=True):
    neigh = neighbors(i, include_center)
    neigh.sort(key=lambda p: img_flatten.value[p], reverse=True)
    return neigh[0]


def freduce_findmin(x, y):
    if img_flatten.value[x] > img_flatten.value[y]:
        return y
    else:
        return x


def fmap_root(x):
    path = x
    root = fmap_argmax_neighbors(int(x))
    while root != path:
        path =root
        root = fmap_argmax_neighbors(int(root))
    return (str(root), x)


def fmap_pts_to_dgms(x):
    x_birth = x[0] // w.value
    y_birth = x[0] % w.value
    x_death = x[1] // w.value
    y_death = x[1] % w.value

    birth = img_flatten.value[x[0]]
    death = img_flatten.value[x[1]]

    return (birth, death, x_birth, y_birth, x_death, y_death)



#Create SparkSession
spark = SparkSession.builder.appName('SparkPH').getOrCreate()


# Load img
img = fits.getdata('f444w_finalV4.onlyPSF.fits').astype(np.float32)[0:1000,0:1000]
img -= img.min()
img /= img.max()


w = spark.sparkContext.broadcast(img.shape[0])
h = spark.sparkContext.broadcast(img.shape[1])
img_flatten = spark.sparkContext.broadcast(img.flatten())


rdd_idxs = spark.sparkContext.parallelize(np.arange(img.size), 256)


rdd_roots = rdd_idxs.map(lambda x: fmap_root(x))
rdd_roots = rdd_roots.reduceByKey(freduce_findmin)

dgms = rdd_roots.collect()

print(len(dgms))
