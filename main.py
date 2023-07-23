# Import libraries
from pyspark.sql import SparkSession
import numpy as np
import pandas as pd
from astropy.io import fits


def neighbors(i, w, h, include_center=False):
    size = w * h
    neighbors = []
    if i - w >= 0:
        neighbors.append(i - w)  # north
    if i % w != 0:
        neighbors.append(i - 1)  # west

    if (i + 1) % w != 0:
        neighbors.append(i + 1)  # east

    if i + w < size:
        neighbors.append(i + w)  # south

    if ((i - w - 1) >= 0) and (i % w != 0):
        neighbors.append(i - w - 1)  # northwest

    if ((i - w + 1) >= 0) and ((i + 1) % w != 0):
        neighbors.append(i - w + 1)  # northeast

    if ((i + w - 1) < size) and (i % w != 0):
        neighbors.append(i + w - 1)  # southwest

    if ((i + w + 1) < size) and ((i + 1) % w != 0):
        neighbors.append(i + w + 1)  # southeast

    if include_center:
        neighbors.append(i)
    return neighbors


def fmap_argmax_neighbors(i, w, h, include_center=False):
    neigh = neighbors(i, w, h, include_center)
    neigh.sort(key=lambda p: img_broad.value[int(p) // w, int(p) % w], reverse=True)
    return neigh[0]


def fmap_root(x):
    path = [x]
    root = parents.value[int(x)]
    while root != path[-1]:
        path.append(root)
        root = parents.value[int(root)]
    return (root, x)


def freduce_findmin(x, y):
    w = img_broad.value.shape[0]
    if img_broad.value[int(x) // w, int(x) % w] > img_broad.value[int(y) // w, int(y) % w]:
        return y
    else:
        return x


def fmap_pts_to_dgms(x):
    w = img_broad.value.shape[0]

    x_birth = x[0] // w
    y_birth = x[0] % w
    x_death = x[1] // w
    y_death = x[1] % w

    birth = img_broad.value[x_birth, y_birth]
    death = img_broad.value[x_death, y_death]

    return (birth, death, x_birth, y_birth, x_death, y_death)



#Create SparkSession
spark = SparkSession.builder.appName('SparkPH').getOrCreate()


# Load img
img = fits.getdata('f444w_finalV4.onlyPSF.fits').astype(np.float32)[0:1000,0:1000]
img -= img.min()
img /= img.max()


img_broad = spark.sparkContext.broadcast(img)
rdd_idxs = spark.sparkContext.parallelize(np.arange(img.size), numSlices=100)


rdd_parents = rdd_idxs.map(lambda x: fmap_argmax_neighbors(x, img_broad.value.shape[0], img_broad.value.shape[1], include_center=True))
parents = spark.sparkContext.broadcast(rdd_parents.collect())

rdd_roots = rdd_idxs.map(lambda x: fmap_root(x))
rdd_roots = rdd_roots.reduceByKey(freduce_findmin)

rdd_dgms = rdd_roots.map(fmap_pts_to_dgms)


dgms = rdd_dgms.collect()#.saveAsTextFile('prova.txt')
dgms = pd.DataFrame(dgms, columns=['birth', 'death', 'x_birth', 'y_birth', 'x_death', 'y_death'])


dgms.to_csv('dgms.csv', index=False)
