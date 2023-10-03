

def neighbors(i, h, w, mode=8):
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

    if mode >= 8:
        if ((i - w - 1) >= 0) and (i % w != 0):
            neighbors.append(i - w - 1)  # northwest

        if ((i - w + 1) >= 0) and ((i + 1) % w != 0):
            neighbors.append(i - w + 1)  # northeast

        if ((i + w - 1) < size) and (i % w != 0):
            neighbors.append(i + w - 1)  # southwest

        if ((i + w + 1) < size) and ((i + 1) % w != 0):
            neighbors.append(i + w + 1)  # southeast
    if mode == 9:
        neighbors.append(i)
    return neighbors