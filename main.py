import numpy as np
from astropy.io import fits

from ph0 import persistence

import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='main',
        description='PH0',
        epilog='')

    parser.add_argument('--image_path', dest='img_path', action='store',
                        default='', help='')

    args = parser.parse_args()

    img = np.load(args.img_path).astype(np.float32)

    dgm = persistence(img)
    print(dgm)

