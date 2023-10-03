import numpy as np
from astropy.io import fits

from ph0 import persistence

import sys

arg1 = sys.argv[1]
arg2 = sys.argv[2]

img = fits.getdata("C:\\Users\\ricca\\Documents\\PyProjects\\TopoDenoising\\data\\f444w_finalV4.onlyPSF.fits")

image = img[0:200,0:200].astype(np.float32)

dgm = persistence(image)

print(dgm)