def my_saddle(x):
    x = x.flatten()
    if x[4] < x[3] and x[4] < x[5]:
        if x[4] > x[1] or x[4] > x[7]:
            return True
        else:
            return False
    elif x[4] < x[1] and x[4] < x[7]:
        if x[4] > x[3] or x[4] > x[5]:
            return True
        else:
            return False
    elif x[4] < x[0] and x[4] < x[8]:
        if x[4] > x[2] or x[4] > x[6]:
            return True
        else:
            return False
    elif x[4] < x[2] and x[4] < x[6]:
        if x[4] > x[0] or x[4] > x[8]:
            return True
        else:
            return False
    else:
        return False


def my_saddle_2(x):
    x = x.flatten()
    if x[4] < x[3] and x[4] < x[5]:
        return True
    elif x[4] < x[1] and x[4] < x[7]:
        return True
    elif x[4] < x[0] and x[4] < x[8]:
        return True
    elif x[4] < x[2] and x[4] < x[6]:
        return True
    else:
        return False