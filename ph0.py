import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

import numpy as np
from utils import maxpool2d
from utils import neighbors
from utils import my_saddle

#from tqdm import tqdm


def persistence(img, return_points=False):
    H, W = img.shape
    p, m = maxpool2d(img, kernel_size=3, stride=1, padding=1, return_indices=True)
    del p

    img = img.flatten()
    m_temp = m.flatten()
    m = m_temp[m_temp]
    while not np.array_equal(m_temp, m):
        m_temp = m
        m = m_temp[m_temp]
    del m_temp

    pbirth = np.unique(m)
    birth = img[pbirth]
    idxs = np.argsort(birth)
    pbirth = pbirth[idxs]

    dict_replace = {pbirth[i]: i+1 for i in range(birth.size)}
    p_idxs = np.sort(idxs + 1)


    def replace(element):
        return dict_replace.get(element, element)
    vreplace = np.vectorize(replace)
    new_m = vreplace(m)

    del birth, idxs, dict_replace, vreplace

    p1 = maxpool2d(new_m.reshape(H, W), kernel_size=3, stride=1, padding=1, return_indices=False).flatten()
    p2 = -maxpool2d(-new_m.reshape(H, W), kernel_size=3, stride=1, padding=1, return_indices=False).flatten()

    borders_idxs = set(np.nonzero(p1 != p2)[0].tolist())

    del p1, p2

    img_pad = np.pad(img.reshape(H,W), ((1, 1), (1, 1)), 'constant', constant_values=0)
    #if img.min() not in img[pbirth]:
    for x in borders_idxs.copy():
        w = x // W
        h = x % W
        if not my_saddle(img_pad[w:(w+3), h:(h+3)]):
            borders_idxs.discard(x)
    for x in np.flip(pbirth):
        for y in neighbors(x, H, W):
            if new_m[y] != new_m[x]:
                borders_idxs.add(y)
        borders_idxs.discard(x)
    del img_pad

    borders_idxs = np.array(list(borders_idxs))
    sort = np.flip(np.argsort(img[borders_idxs]))
    borders_idxs = borders_idxs[sort]

    pdeath = []
    death_idxs = set()
    changer = p_idxs.copy()
    #for x in tqdm(borders_idxs):
    for x in borders_idxs:
        # TODO: se tutti i death point sono trovati allora fermati!!
        if len(pdeath) == (len(pbirth) - 1):
            break
        check = np.unique(changer[new_m[np.array(neighbors(x, H, W, mode=9))] - 1])
        if len(check) >= 2:
            check = tuple(check[:2])
            if check not in death_idxs:
                death_idxs.add(check)
                changer[np.where(changer == check[0])[0]] = check[1]
                pdeath.append([check[0] - 1, x])
    del changer

    pdeath = np.array(pdeath)
    pdeath = pdeath[np.argsort(pdeath[:, 0])][:, 1]
    pdeath = np.append(pdeath, np.argmin(img))

    if return_points:
        dgm = np.stack([img[pbirth],
                        img[pdeath],
                        pbirth % W,
                        pbirth // W,
                        pdeath % W,
                        pdeath // W], axis=1)
    else:
        dgm = np.stack([img[pbirth],
                        img[pdeath]], axis=1)

    return dgm