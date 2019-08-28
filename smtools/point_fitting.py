#!/usr/bin/env python
\

__all__ = ['find_maxima', 'fit_routine']

__version__ = "0.3.0"

__author__ = 'Sy Redding'



import numpy as np
import skimage.filters as skim
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from scipy.optimize import curve_fit
from scipy.spatial.distance import cdist
from smtools.misc import two_d_gaussian



def find_maxima(image,size,threshold_method = "threshold_mean"):
    """
    Locates maxima in an image. See `here
    <https://github.com/ReddingLab/Learning/blob/master/image-analysis-basics/2__finding-local-maxima.ipynb/>`_

    :param image: 2-dimensional image array.
    :param size: int, size of the maxima and minimum filters used
    :param threshold_method: string, type of thresholding filter
        used. Accepts any threshold_filter in the ``skimmage.filters``
        module.
        Default is threshold_mean

    :return: 1D array of [(x,y),...] defining the locations of each
        maximum

    :Example:
        >>> import smtools.point_fitting as pt
        >>> import smtools.testdata as test
        >>> im = test.single_max()
        >>> print(pt.find_maxima(im,10))
        [(17, 14)]
    """
    im_max = filters.maximum_filter(image, size)
    im_min = filters.minimum_filter(image, size)
    im_diff = im_max - im_min

    maxima=(image==im_max)
    thresh = getattr(skim, threshold_method)(im_diff)
    bool_diff = (im_diff < thresh)
    maxima[bool_diff] = False

    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)
    points = []
    for dy,dx in slices:
        points.append((dx.start,dy.start)) 
    return points

def fit_routine(Image, points, bbox, err = 2,
                frame = 1, minimum_separation = 1.5):
    """
    Fits a 2D gaussian function to 2D image array. "x", "y",
        and "bbox" define an ROI fit by ``two_d_gaussian`` using
        ``scipy.optimaize.curve_fit``

    :param Image: 2D array containing ROI to be fit
    :param points: 1D array of (x,y) points as tuples. accepts output
        directly from ``find_maxima``
    :param bbox: side-length of the ROI. if even, rounds up to next
        odd integer
    :param err: int or float. Threshold for the error on the fit of the
        Gaussian mean. Applied to both the mean in x and y.
    :param frame: int or float. Frame number or time value
    :param minimum_separation: int or float. minimum distance between
        two fitted points

    :return: 1D array of optimal (x,y) positions from gaussian fit.
        If ROI falls (partially or otherwise) outside Image,
        or, if curve_fit raises RuntimeError or ValueError,
        or, if the error threshold is crossed, then that maxima is
        passed over.

    :Example:
        >>> import smtools.point_fitting as pt
        >>> import smtools.testdata as test
        >>> im = test.single_max()
        >>> x,y = pt.find_maxima(im,10)[0]
        >>> fit = pt.fit_routine(im, [(x, y)], 10)
        >>> print(Fit)
        array([ 16.74464408, 14.28216362 ])
    """
    bbox = bbox+(1-bbox%2)
    db = int(np.floor(bbox/2))

    fits = []
    for x,y in points:
        span_x = np.linspace(x-db,x+db, bbox)
        span_y = np.linspace(y-db,y+db, bbox)
        X,Y = np.meshgrid(span_x, span_y)
        if 0<= y-db <= y+db+1 <= Image.shape[0] and 0<= x-db <= x+db+1 <= Image.shape[1]:
            pixel_vals = Image[y-db:y+db+1, x-db:x+db+1].ravel()
            scaled = [k/max(pixel_vals) for k in pixel_vals]
            initial_guess = (1, x, y, 1, 1, 0, 0)
            try:
                popt, pcov = curve_fit(two_d_gaussian, (X, Y), scaled,
                                       p0=initial_guess)

                if (np.sqrt(pcov[1][1]) < err and
                    np.sqrt(pcov[2][2]) < err):
                    fits.append((popt[1],popt[2]))
                else:
                    fits.append(None)
            except (RuntimeError, ValueError):
                fits.append(None)
        else:
            fits.append(None)

    #-- filtering points by proximity
    dist_arr = np.tril(cdist(fits, fits, 'euclidean'))
    arr = np.where((dist_arr < minimum_separation) & (
            dist_arr > 0))
    indexes = list(set(np.concatenate((arr[0], arr[1]))))
    for index in sorted(indexes, reverse=True):
        del fits[index]

    fits = [i + (frame,) for i in fits]
    return(fits)
