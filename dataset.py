#!/usr/bin/env python
# coding: utf-8


import numpy as np
from tqdm.auto import tqdm
from astropy.modeling import models
from datetime import datetime
from scipy.ndimage import gaussian_filter

import argparse
import os


def read_noise(image, amount, gain=1):
    """
    Generate simulated read noise.
    
    Parameters
    ----------
    
    image: numpy array
        Image whose shape the noise array should match.
    amount : float
        Amount of read noise, in electrons.
    gain : float, optional
        Gain of the camera, in units of electrons/ADU.
    """
    shape = image.shape
    
    noise = noise_rng.normal(scale=amount/gain, size=shape)
    
    return noise


def bias(image, value, realistic=False):
    """
    Generate simulated bias image.
    
    Parameters
    ----------
    
    image: numpy array
        Image whose shape the bias array should match.
    value: float
        Bias level to add.
    realistic : bool, optional
        If ``True``, add some columns with somewhat higher bias value (a not uncommon thing)
    """
    # This is the whole thing: the bias is really suppose to be a constant offset!
    bias_im = np.zeros_like(image) + value
    
    # If we want a more realistic bias we need to do a little more work. 
    if realistic:
        shape = image.shape
        number_of_colums = 5
        
        # We want a random-looking variation in the bias, but unlike the readnoise the bias should 
        # *not* change from image to image, so we make sure to always generate the same "random" numbers.
        rng = np.random.RandomState(seed=8392)  # 20180520
        columns = rng.randint(0, shape[1], size=number_of_colums)
        # This adds a little random-looking noise into the data.
        col_pattern = rng.randint(0, int(0.1 * value), size=shape[0])
        
        # Make the chosen columns a little brighter than the rest...
        for c in columns:
            bias_im[:, c] = value + col_pattern
            
    return bias_im


def dark_current(image, current, exposure_time, gain=1.0, hot_pixels=False):
    """
    Simulate dark current in a CCD, optionally including hot pixels.
    
    Parameters
    ----------
    
    image : numpy array
        Image whose shape the cosmic array should match.
    current : float
        Dark current, in electrons/pixel/second, which is the way manufacturers typically 
        report it.
    exposure_time : float
        Length of the simulated exposure, in seconds.
    gain : float, optional
        Gain of the camera, in units of electrons/ADU.
    strength : float, optional
        Pixel count in the cosmic rays.    
    """
    
    # dark current for every pixel; we'll modify the current for some pixels if 
    # the user wants hot pixels.
    base_current = current * exposure_time / gain
    
    # This random number generation should change on each call.
    dark_im = noise_rng.poisson(base_current, size=image.shape)
        
    if hot_pixels:
        # We'll set 0.01% of the pixels to be hot; that is probably too high but should 
        # ensure they are visible.
        y_max, x_max = dark_im.shape
        
        n_hot = int(0.0001 * x_max * y_max)
        
        # Like with the bias image, we want the hot pixels to always be in the same places
        # (at least for the same image size) but also want them to appear to be randomly
        # distributed. So we set a random number seed to ensure we always get the same thing.
        rng = np.random.RandomState(16201649)
        hot_x = rng.randint(0, x_max, size=n_hot)
        hot_y = rng.randint(0, y_max, size=n_hot)
        
        hot_current = 10000 * current
        
        dark_im[(hot_y, hot_x)] = hot_current * exposure_time / gain
    return dark_im


def sky_background(image, sky_counts, gain=1):
    """
    Generate sky background, optionally including a gradient across the image (because
    some times Moons happen).
    
    Parameters
    ----------
    
    image : numpy array
        Image whose shape the cosmic array should match.
    sky_counts : float
        The target value for the number of counts (as opposed to electrons or 
        photons) from the sky.
    gain : float, optional
        Gain of the camera, in units of electrons/ADU.
    """
    sky_im = noise_rng.poisson(sky_counts * gain, size=image.shape) / gain
    
    return sky_im

def stars(image, number, max_counts=10000, gain=1, dim_max_star=250, seed=12345):
    """
    Add some stars to the image.
    """
    from photutils.datasets import make_random_gaussians_table, make_gaussian_sources_image
    # Most of the code below is a direct copy/paste from
    # https://photutils.readthedocs.io/en/stable/_modules/photutils/datasets/make.html#make_100gaussians_image
    
    flux_range = [max_counts/10, max_counts]
    
    y_max, x_max = image.shape
    xmean_range = [int(dim_max_star / 2), x_max + int(dim_max_star / 2)]
    ymean_range = [int(dim_max_star / 2), y_max + int(dim_max_star / 2)]
    xstddev_range = [4, 4]
    ystddev_range = [4, 4]
    params = dict([('amplitude', flux_range),
                  ('x_mean', xmean_range),
                  ('y_mean', ymean_range),
                  ('x_stddev', xstddev_range),
                  ('y_stddev', ystddev_range),
                  ('theta', [0, 2*np.pi])])

    sources = make_random_gaussians_table(number, params,
                                          seed=seed)

    model = models.Gaussian2D(x_stddev=1, y_stddev=1)

    H, W = image.shape

    image = np.zeros((H + 2 * dim_max_star, W + 2 * dim_max_star), dtype=float)
    
    params_to_set = []
    for param in sources.colnames:
        if param in model.param_names:
            params_to_set.append(param)

    init_params = {param: getattr(model, param) for param in params_to_set}
    
    #star_im = make_gaussian_sources_image(image.shape, sources)
    
    for source in tqdm(sources):
        for param in params_to_set:
            setattr(model, param, source[param])
            
        yidx, xidx = np.indices((int(dim_max_star),int(dim_max_star)))

        xidx = xidx + int(source['x_mean']) - int(dim_max_star / 2)
        yidx = yidx + int(source['y_mean']) - int(dim_max_star / 2)

        image[(int(source['x_mean']) - int(dim_max_star / 2)):(int(source['x_mean']) + int(dim_max_star / 2)), (int(source['y_mean']) - int(dim_max_star / 2)):(int(source['y_mean']) + int(dim_max_star / 2))] = image[(int(source['x_mean']) - int(dim_max_star / 2)):(int(source['x_mean']) + int(dim_max_star / 2)), (int(source['y_mean']) - int(dim_max_star / 2)):(int(source['y_mean']) + int(dim_max_star / 2))] + model(xidx, yidx)
    
    
    return image[dim_max_star:(dim_max_star + H),dim_max_star:(dim_max_star + W)]



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='dataset',
        description='Create astronomical image dataset',
        epilog='')

    parser.add_argument('-o', dest='out_dir', action='store',
                        default='dataset', help='')
    parser.add_argument('-dd', dest='dataset_dim', action='store',
                        default=1, help='')
    parser.add_argument('-id', dest='image_dim', action='store',
                        default=10000, help='')

    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    for i in range(int(args.dataset_dim)):

        seed = int(datetime.now().timestamp())
        noise_rng = np.random.default_rng(seed)

        synthetic_image = np.zeros([int(args.image_dim), int(args.image_dim)])
        noise_only = read_noise(synthetic_image, amount=5, gain=1)
        bias_only = bias(synthetic_image, 1100, realistic=True)

        dark_exposure = 100
        dark_cur = 0.1
        dark_only = dark_current(synthetic_image, dark_cur, dark_exposure, hot_pixels=True)

        sky_level = 5
        sky_only = sky_background(synthetic_image, sky_level)

        stars_only = stars(synthetic_image, int(0.0034 * synthetic_image.size), max_counts=2000, seed=seed) #

        stars_with_background = synthetic_image + stars_only + noise_only + sky_only + bias_only + dark_only

        img = stars_with_background #gaussian_filter(stars_with_background, sigma=5)
        np.save(args.out_dir + '/{}.npy'.format(i+1),img)
