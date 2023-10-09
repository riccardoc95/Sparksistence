import numpy as np
import argparse

import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


from ph0 import persistence


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='main',
        description='PH0',
        epilog='')

    parser.add_argument('--image_path', dest='img_path', action='store',
                        default='', help='')

    args = parser.parse_args()

    img = np.load(args.img_path).astype(np.float32, copy=False)

    dgm = persistence(img)
    print(dgm)

