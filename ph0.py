import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

import numpy as np
from utils import maxpool2d
from utils import neighbors
from utils import my_saddle


def persistence(img, return_points=False):
    img -= img.min()
    img /= img.max()

    img_min = 0.0

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

    # Mod in new algorithm version:
    m = m * (img > 0) + img.argmin() * (img == 0)

    pbirth = np.unique(m)
    birth = img[pbirth]
    idxs = np.argsort(birth)
    pbirth = pbirth[idxs]

    dict_replace = {pbirth[i]: i + 1 for i in range(birth.size)}
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

    pbirth = pbirth[1:]

    img_pad = np.pad(img.reshape(H,W), ((1, 1), (1, 1)), 'constant', constant_values=0)


    for x in borders_idxs.copy():
        w = x // W
        h = x % W
        if img[x] == img_min:
            continue
        elif img_min in img[neighbors(x, H, W, mode=8)]:
            borders_idxs.discard(x)
        elif not my_saddle(img_pad[w:(w + 3), h:(h + 3)]):
            borders_idxs.discard(x)
    for x in np.flip(pbirth):
        for y in neighbors(x, H, W):
            if new_m[y] != new_m[x] and new_m[x] != 0.0:
                borders_idxs.add(y)
        borders_idxs.discard(x)
    del img_pad

    borders_idxs = np.array(list(borders_idxs))
    sort = np.flip(np.argsort(img[borders_idxs]))
    borders_idxs = borders_idxs[sort]

    pdeath = []
    death_idxs = set()
    # Mod in new algorithm version:

    changer = p_idxs.copy()
    changer[0] = changer[-1]


    len_pbirth = (len(pbirth) - 1)

    for x in borders_idxs:
        # Mod in new algorithm version:
        if len(pdeath) == len_pbirth:
            break
        check = np.unique(changer[new_m[np.array(neighbors(x, H, W, mode=9))] - 1])
        if len(check) >= 2:
            check = tuple(check[:2])
            if check not in death_idxs:
                death_idxs.add(check)
                changer[np.where(changer == check[0])[0]] = check[1]
                pdeath.append([check[0] - 1, x])
    del changer

    if len(pdeath) > 0:
        pdeath = np.array(pdeath)
        pdeath = pdeath[np.argsort(pdeath[:, 0])][:, 1]
        pdeath = np.append(pdeath, np.argmin(img))
    else:
        pdeath = np.array([np.argmin(img)])

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