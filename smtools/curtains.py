#!/usr/bin/env python
from itertools import combinations

__all__ = ['find_rotang', 'find_curtain',
           'find_DNA', 'fit_DNA']

__version__ = '0.3.0'

__author__ = 'Sy Redding and Luke Strauskulage'

########################################################################
from smtools.misc import super_gauss_function, mundane_gauss_function
from skimage.feature import canny
from skimage.transform import probabilistic_hough_line
from skimage.exposure import equalize_adapthist
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit
from scipy.spatial import distance
from numpy.linalg import norm

from itertools import combinations
from sklearn.cluster import KMeans
from scipy.stats import norm as scinorm

import numpy as np
import warnings


def find_rotang(Image, line_length=40,
                theta=None, line_gap=5, tilt=.1):
    """
    Finds the general rotation of an image containing DNA curtains.
    Uses a
    `Hough line Transform
    <http://scikit-image.org/docs/dev/api/skimage.transform.html
    #skimage.transform.probabilistic_hough_line>`_
    to approximate DNA locations. Returns the average angle of all
    hough lines.

    :param Image: 2D image array
    :param line_length: int, minimum accepted length of line detected
        by the `Hough line Transform
        <http://scikit-image.org/docs/dev/api/skimage.transform.html
        #skimage.transform.probabilistic_hough_line>`_
        Default is 40.
    :param theta: array, angles at which to compute the
        `Hough line Transform
        <http://scikit-image.org/docs/dev/api/skimage.transform.html
        #skimage.transform.probabilistic_hough_line>`_
        in radians. Default is None
    :param line_gap: int, maximum gap between pixels allowed when
        forming a line. default is 5
    :param tilt: int, when theta is None, the tilt value defines the
        range of thetas to be used in the
        `Hough line Transform
        <http://scikit-image.org/docs/dev/api/skimage.transform.html
        #skimage.transform.probabilistic_hough_line>`_
        as (pi/2 - tilt:pi/2 + tilt). Defaults to 0.1 radians

    :return: measure rotation angle in radians:

    :Example:
        >>> import smtools.curtain as cs
        >>> import smtools.testdata as test
        >>> im = test.test_curtain()
        >>> ch1,ch2 = al.im_split(i)
        >>> print(cs.find_rotang(ch2))
        0.6936316728744394
    """

    if theta is None:
        theta = np.linspace(np.pi / 2. - tilt, np.pi / 2. + tilt, 500)

    edges = canny(equalize_adapthist(Image))
    lines = probabilistic_hough_line(edges, line_length=line_length,
                                     theta=theta, line_gap=line_gap)

    rise = [line[1][1] - line[0][1] for line in lines]
    run = [line[1][0] - line[0][0] for line in lines]
    rotation = 180 * np.tan(np.mean(rise) / np.mean(run)) / np.pi
    return rotation


def find_curtain(Image, distance=50, line_length=40,
                 theta=None, line_gap=5, tilt=.1,
                 window=15, order=3, maxline=70):
    """

    :param Image: 2D image array
    :param distance: int, passes to
        `peak_finder <https://docs.scipy.org/doc/scipy/reference
        /generated/scipy.signal.find_peaks.html>`_.
        defines minimum spacing between two separate curtains.
        Default is 50.
    :param line_length: int, minimum accepted length of line detected
        by the `Hough line Transform
        <http://scikit-image.org/docs/dev/api/skimage.transform.html
        #skimage.transform.probabilistic_hough_line>`_
        Default is 40.
    :param theta: array, angles at which to compute the
        `Hough line Transform
        <http://scikit-image.org/docs/dev/api/skimage.transform.html
        #skimage.transform.probabilistic_hough_line>`_
        in radians. Default is None
    :param line_gap: int, maximum gap between pixels allowed when
        forming a line. default is 5
    :param tilt: int, when theta is None, the tilt value defines the
        range of thetas to be used in the
        `Hough line Transform
        <http://scikit-image.org/docs/dev/api/skimage.transform.html
        #skimage.transform.probabilistic_hough_line>`_
        as (pi/2 - tilt:pi/2 + tilt). Defaults to 0.1 radians
    :param window: int, window value passed to the
        `savgol_filter <https://docs.scipy.org/doc/scipy-0.15.1
        /reference/generated/scipy.signal.savgol_filter.html>`_
    :param order: int, order value passed to the
        `savgol_filter <https://docs.scipy.org/doc/scipy-0.15.1
        /reference/generated/scipy.signal.savgol_filter.html>`_
    :param maxline: int, maximum size line detected by the
        `Hough line Transform
        <http://scikit-image.org/docs/dev/api/skimage.transform.html
        #skimage.transform.probabilistic_hough_line>`_.
        defaults to 70.

    :return: List of tuples containing (x_min, x_max, y_min, y_max)
        for each curtain in the image.
    :return: Boolean mask of the image. Evaluates True within the
        bounds of the detected curtains, and False outside.

    :Example:
        >>> import smtools.curtain as cs
        >>> import smtools.testdata as test
        >>> from scipy.ndimage.interpolation import rotate
        >>> im = test.test_curtain()
        >>> ch1,ch2 = al.im_split(i)
        >>> angle = cs.find_rotang(ch2)
        >>> rotated_ch2 = rotate(ch2,angle)
        >>> bounds, mask = cs.find_curtain(rotated_ch2)
        >>> print(bounds)
        [(1, 76, 0, 515), (89, 157, 0, 513), (167, 235, 0, 515)]

    """
    if theta is None:
        theta = np.linspace(np.pi / 2. - tilt, np.pi / 2. + tilt, 500)

    edges = canny(equalize_adapthist(Image))
    lines = probabilistic_hough_line(edges, line_length=line_length,
                                     theta=theta, line_gap=line_gap)

    line_starts = [line[0][0] for line in lines]
    line_ends = [line[1][0] for line in lines]
    x = np.linspace(0, Image.shape[1] + 1, Image.shape[1] + 1)
    starts, bins = np.histogram(line_starts, bins=x)
    ends, bins = np.histogram(line_ends, bins=x)
    smoothed_starts = savgol_filter(starts, window, order)
    smoothed_ends = savgol_filter(ends, window, order)
    peaks_start, _ = find_peaks(smoothed_starts, distance=distance)
    peaks_end, _ = find_peaks(smoothed_ends, distance=distance)
    curtains = []
    for j in peaks_start:
        for k in peaks_end:
            if (k - j > line_length
                    and k - j < line_length + 20
                    and smoothed_starts[j] > 1.
                    and smoothed_ends[k] > 1.):
                curtains.append((j, k))
    d = {key[0]: [] for key in curtains}
    for line in lines:
        key = min(peaks_start, key=lambda x: abs(x - line[0][0]))
        if line[1][0] - line[0][0] < maxline:
            try:
                d[key].append(line[0][1])
            except KeyError:
                pass
    data = []
    curtain_mask = np.ones_like(Image, dtype=bool)
    for m, n in zip(d, curtains):
        x_min = min(np.clip([n[0] - 10], 0, Image.shape[1]))
        x_max = max(np.clip([n[1] + 10], 0, Image.shape[1]))
        y_min = min(
            np.clip([x - 5 for x in d[m]], 0, Image.shape[0]))
        y_max = max(
            np.clip([x + 5 for x in d[m]], 0, Image.shape[0]))

        data.append((x_min, x_max, y_min, y_max))
        curtain_mask[y_min:y_max, x_min:x_max] = False

    return data, curtain_mask


def find_DNA(Image, Bounds, prominence=None):
    """

    :param Image: 2D image array
    :param Bounds: 1D List of tuples containing
        (x_min, x_max, y_min, y_max) for each curtain in the image.
        written to accept output from toolbox.curtains.find_curtain
    :param prominence: passes to
        `peak_finder
        <https://docs.scipy.org/doc/scipy/reference/generated/scipy
        .signal.find_peaks.html>`_
        Defaults to (20, 1.5 * max(array)

    :return: List of tuples: (top, bottom, and center). Locations of
        each DNA strand detected in pixels.
    :Example:
        >>> import smtools.curtain as cs
        >>> import smtools.testdata as test
        >>> from scipy.ndimage.interpolation import rotate
        >>> im = test.test_curtain()
        >>> ch1,ch2 = al.im_split(i)
        >>> angle = cs.find_rotang(ch2)
        >>> rotated_ch2 = rotate(ch2,angle)
        >>> bounds, mask = cs.find_curtain(rotated_ch2)
        >>> strands = cs.find_DNA(rotated_ch2,bounds)
        >>> print(len(strands))
        269
    """
    data = []
    point_im = np.zeros_like(Image, dtype=bool)
    for i in range(Image.shape[1]):
        subim = Image[:, i]
        peaks, properties = find_peaks(subim)
        for j in peaks:
            point_im[j, i] = True
    for i in Bounds:
        subarr = point_im[i[2]:i[3], i[0]:i[1]]
        flattened_arr = [sum(subarr[j, :]) for j in range(len(subarr))]
        if prominence is None:
            prominence = (20, 1.5 * max(flattened_arr))
        peaks, properties = find_peaks(flattened_arr, distance=3,
                                       prominence=prominence)
        for j in peaks:
            data.append((i[0], i[1], i[2] + j))
    return data


def fit_DNA(Image, locations, constraints=None,
            init_guess_length=None, init_guess_center=None,
            pad=2, factor=1.3, min_length=45, max_length=56,
            cen_err=.3, cen_dev=3.):
    """

    :param Image: 2D image array
    :param locations: 1List of tuples: (top, bottom, and center).
        Locations of each DNA strand detected in pixels.
        written to accept output from toolbox.curtains.find_DNA
    :param constraints: 1D list, size 5. bounds on the call to
        curve_fit when fitting toolbox.misc.super_gauss_function. see
        `bounds <https://docs.scipy.org/doc/scipy/reference/generated
        /scipy.optimize.curve_fit.html>`_
        Default is ([0, 0, 0, 0, 5.],[np.inf, np.inf, np.inf, np.inf,
        np.inf]).
    :param init_guess_length: 1D list, size 5. initial guess for the
        parameters for the call to curve_fit when fitting
        toolbox.misc.super_gauss_function. see
        `p0 <https://docs.scipy.org/doc/scipy/reference/generated
        /scipy.optimize.curve_fit.html>`_
        Default is [.2, 1., np.median(xdata), 20, 6].
    :param init_guess_center: 1D list, size 5. initial guess for the
        parameters for the call to curve_fit when fitting
        toolbox.misc.mundane_gauss_function. see
        `p0 <https://docs.scipy.org/doc/scipy/reference/generated
        /scipy.optimize.curve_fit.html>`_
        Default is [1, 1, 2, 1].
    :param pad: int, number of adjacent rows of pixels to grab for
        the fit, Default is 2 pixels out on each side.
    :param factor: float, multiplier used in end detection. defines
        the intensity threshold at the ends as
        maximum - ((maximum - minimum)/factor)
    :param min_length: int, minimum allowed length of DNA.
        Default is 45.
    :param max_length: int, maximum allowed length of DNA.
        Default is 56.
    :param cen_err: float, maximum error ratio on the mean for the
        call to
        `curve_fit <https://docs.scipy.org/doc/scipy/reference
        /generated/scipy.optimize.curve_fit.html>`_
        when fitting toolbox.misc.mundane_gauss_function.
        Defined as np.sqrt(pcov[2][2])/popt[2]
        Defualts to 0.3.
    :param cen_dev: float, maximum allowed standard deviation for the
        call to
        `curve_fit <https://docs.scipy.org/doc/scipy/reference
        /generated/scipy.optimize.curve_fit.html>`_
        when fitting toolbox.misc.mundane_gauss_function.
        Defualts to 3.

    :return: List of tuples: (top, bottom, and center). Locations of
        each DNA strand detected as defined by the fitting with
        toolbox.misc.super_gauss_function and
        toolbox.misc.mundane_gauss_function.

    :Example:
        >>> import smtools.curtain as cs
        >>> import smtools.testdata as test
        >>> from scipy.ndimage.interpolation import rotate
        >>> im = test.test_curtain()
        >>> ch1,ch2 = al.im_split(i)
        >>> angle = cs.find_rotang(ch2)
        >>> rotated_ch2 = rotate(ch2,angle)
        >>> bounds, mask = cs.find_curtain(rotated_ch2)
        >>> strands = cs.find_DNA(rotated_ch2,bounds)
        >>> DNAs = cs.fit_DNA(rotated_ch2, strands)
        >>> print(len(DNAs))
        200
        >>> import matplotlib.pyplot as plt
        >>> plt.figure()
        >>> plt.imshow(rotated_ch2)
        >>> for x0,x1,y in DNAs:
        >>>     plt.plot([x0,x1],[y,y],"r.", markersize = 5)
        >>> plt.show()
    """
    warnings.filterwarnings('ignore')
    if constraints is None:
        constraints = ([0, 0, 0, 0, 5.],
                       [np.inf, np.inf, np.inf, np.inf, np.inf])
    output = []
    for i in locations:
        if i[2] - pad < 0 or i[2] + pad + 1 > Image.shape[0]:
            pass
        else:
            local_dna = Image[i[2] - pad:i[2] + pad + 1,
                        i[0]:i[1]]
            # ----------------------------------------------------#
            # -----------------fit the DNA length-----------------#
            # ----------------------------------------------------#
            average_intensity = local_dna.mean(axis=0)
            normed_dna = np.array(
                [float(value) / max(average_intensity) for value in
                 average_intensity])
            xdata = np.array([j for j in range(len(normed_dna))])
            if init_guess_length is None:
                init_guess_length = [.2, 1., np.median(xdata), 20, 6]
            try:
                popt, pcov = curve_fit(super_gauss_function, xdata,
                                       normed_dna,
                                       p0=init_guess_length,
                                       bounds=constraints)
                fine_scale_x = np.linspace(xdata[0], xdata[-1],
                                           len(xdata) * 1000)
                maximum = max(super_gauss_function(fine_scale_x, *popt))
                intersection_value = maximum - (
                        maximum - popt[0]) / factor
                linedata = [intersection_value for x in fine_scale_x]
                index = np.argwhere(np.diff(np.sign(
                    linedata - super_gauss_function(fine_scale_x,
                                                    *popt)))).flatten()
                if len(index) == 2:
                    if min_length <= fine_scale_x[index[1]] - fine_scale_x[
                        index[0]] <= max_length:
                        top = i[0] + fine_scale_x[index[0]]
                        bottom = i[0] + fine_scale_x[index[1]]
                    else:
                        top = None
                        bottom = None
                else:
                    top = None
                    bottom = None
            except RuntimeError:
                top = None
                bottom = None

            # ----------------------------------------------------#
            # -----------------fit the DNA center-----------------#
            # ----------------------------------------------------#
            average_intensity = local_dna.mean(axis=1)
            normed_dna = np.array(
                [float(value) / max(average_intensity) for value in
                 average_intensity])
            xdata = np.array([j for j in range(len(normed_dna))])
            if init_guess_center is None:
                init_guess_center = [1, 1, 2, 1]
            try:
                popt, pcov = curve_fit(mundane_gauss_function, xdata,
                                       normed_dna,
                                       p0=init_guess_center)
                if np.sqrt(pcov[2][2]) / popt[2] < cen_err and popt[
                    3] < cen_dev:
                    center = i[2] - 2 + popt[2]
                else:
                    center = None
            except RuntimeError:
                center = None
            if all((top, bottom, center)):
                output.append(((top, center), (bottom, center)))
    return output




class DNA(object):

    def __init__(self, params, id):
        """
        :param params:
        """

        self.id = id

        self.params = params

        self.top = np.array(self.params[0])
        self.bottom = np.array(self.params[1])
        self.extension = norm(self.bottom - self.top)

        self.assigned_points = []
        self.pixels_from_end = []
        self.scaled_points = []

        self.points_targets = []
        self.strand_extension = []
        self.points_pairwise = []
        self.pairs_used = []

    def assign_points(self, points, max_offset=1):
        """

        :param points:
        :param max_offset:
        :return:
        """
        for x, y, t in points:
            d = np.cross(self.bottom - self.top, self.top - np.array(
                (x, y))) / self.extension
            if (self.top[0] <= x <= self.bottom[0] and max_offset >=
                    abs(d)):
                self.assigned_points.append((x, y, t))

    def distance_from_end(self):
        """

        :return:
        """
        for x, y, t in self.assigned_points:
            d1 = norm(self.bottom - np.array((x, y)))
            d2 = np.cross(self.bottom - self.top,
                          self.top - np.array((x, y))) / self.extension
            x_FromEnd = np.sqrt((d1 ** 2) - (d2 ** 2))
            self.pixels_from_end.append((x_FromEnd, y, t))

    def scale_to_bp(self, length=48502):
        """

        :param length:
        :return:
        """
        for x, y, t in self.assigned_points:
            d1 = norm(np.array((x, y)) - self.top)
            d2 = np.cross(self.bottom - self.top,
                          self.top - np.array((x, y))) / self.extension
            d3 = np.sqrt((d1 ** 2) - (d2 ** 2))
            scaled_x = d3 / self.extension * length
            self.scaled_points.append((scaled_x, y, t))

    def binding_count(self):
        """

        :return:
        """
        return len(self.assigned_points)

    def binding_data(self):
        """

        :return:
        """
        if len(self.pixels_from_end):
            return [i[0] for i in self.pixels_from_end]
        elif len(self.scaled_points):
            return [i[0] for i in self.scaled_points]
        elif len(self.assigned_points):
            return [i[0] for i in self.assigned_points]
        else:
            return None

    def pair_with_target(self, pop_params):
        for x, y, t in self.pixels_from_end:
            match = min(pop_params, key = lambda stat: abs(stat[0] - x))
            pixelmean = match[0]
            pixelstdev = match [1]
            bptarget = match[2]
            if x > (pixelmean + abs(1*pixelstdev)) or x < (pixelmean - abs(1*pixelstdev)):
                pass
            else:
                self.points_targets.append((x, bptarget))

    def matched_events(self):
        if len(self.points_targets):
            return [x for x in self.points_targets]
        else:
            return None

    def remove_population(self, target, max_target):
        for each in self.points_targets:
            if each[1] == target:
                self.points_targets.remove(each)
            elif each[1] > max_target:
                self.points_targets.remove(each)
            else:
                pass

    def position_pairwise(self, excluded1):
        if len(self.points_targets) > 2:
            pairs = [each for each in combinations(self.points_targets, 2)]
            for twoevents in pairs:
                if twoevents[0] > twoevents[1]:
                    event1 = twoevents[0]
                    event2 = twoevents[1]
                else:
                    event1 = twoevents[1]
                    event2 = twoevents[0]
                if event1[1] == event2[1]:
                    pass
                elif event1[1] == excluded1 or event2[1] == excluded1:
                    pass
                else:
                    barcode = (event1[1] + event2[1])
                    pixel_distance = abs(event1[0] - event2[0])
                    basepair_distance = abs(event1[1] - event2[1])
                    extension = basepair_distance / pixel_distance
                    self.strand_extension.append((event1[1], event2[1], barcode,
                                                  pixel_distance, basepair_distance, extension))
                    for x, bptarget in self.points_targets:
                        if x == event1[0] or x == event2[0]:
                            pass
                        else:
                            relative_distance = abs(x - event1[0])
                            basepair_difference = relative_distance * extension
                            if x > event1[0]:
                                measured_position = event1[1] - basepair_difference
                                self.points_pairwise.append((x, bptarget, measured_position,
                                                             twoevents, barcode))
                            else:
                                measured_position = event1[1] + basepair_difference
                                self.points_pairwise.append((x, bptarget, measured_position,
                                                             twoevents, barcode))

    def filtered_pairwise(self, pair_params, excluded1, excluded2):
        if len(self.points_targets) > 2:
            pairs = [each for each in combinations(self.points_targets, 2)]
            for twoevents in pairs:
                if twoevents[0] > twoevents[1]:
                    event1 = twoevents[0]
                    event2 = twoevents[1]
                else:
                    event1 = twoevents[1]
                    event2 = twoevents[0]
                if event1[1] == event2[1]:
                    pass
                elif event1[1] == excluded1 or event2[1] == excluded1:
                    pass
                elif event1[1] == excluded2 or event2[1] == excluded2:
                    pass
                else:
                    barcode = (event1[1] + event2[1])
                    for stats in pair_params:
                        if barcode == stats[0]:
                            pixel_distance = abs(event1[0] - event2[0])
                            basepair_distance = abs(event1[1] - event2[1])
                            extension = basepair_distance / pixel_distance
                            if extension > stats[1] + 1*(abs(stats[2])) or extension < stats[1] - 1*(abs(stats[2])):
                                pass
                            else:
                                self.strand_extension.append((event1[1], event2[1], barcode,
                                                              pixel_distance, basepair_distance, extension))
                                for x, bptarget in self.points_targets:
                                    if x == event1[0] or x == event2[0]:
                                        pass
                                    else:
                                        relative_distance = abs(x - event1[0])
                                        basepair_difference = relative_distance * extension
                                        if x > event1[0]:
                                            measured_position = event1[1] - basepair_difference
                                            calc_error = measured_position - bptarget
                                            self.points_pairwise.append((x, bptarget, measured_position,
                                                                         relative_distance, calc_error,
                                                                         twoevents, barcode))
                                        else:
                                            measured_position = event1[1] + basepair_difference
                                            calc_error = measured_position - bptarget
                                            self.points_pairwise.append((x, bptarget, measured_position,
                                                                         relative_distance, calc_error,
                                                                         twoevents, barcode))
                        else:
                            pass

    def filtered_flanking(self, pair_params, excluded1):
        if len(self.points_targets) > 2:
            pairs = [each for each in combinations(self.points_targets, 2)]
            for twoevents in pairs:
                if twoevents[0] > twoevents[1]:
                    event1 = twoevents[0]
                    event2 = twoevents[1]
                else:
                    event1 = twoevents[1]
                    event2 = twoevents[0]
                if event1[1] == event2[1]:
                    pass
                elif event1[1] == excluded1 or event2[1] == excluded1:
                    pass
                else:
                    barcode = (event1[1] + event2[1])
                    for stats in pair_params:
                        if barcode == stats[0]:
                            pixel_distance = abs(event1[0] - event2[0])
                            basepair_distance = abs(event1[1] - event2[1])
                            extension = basepair_distance / pixel_distance
                            if extension > stats[1] + 1*(abs(stats[2])) or extension < stats[1] - 1*(abs(stats[2])):
                                pass
                            else:
                                for x, bptarget in self.points_targets:
                                    if x == event1[0] or x == event2[0]:
                                        pass
                                    if event1[0] < x < event2[0] or event2[0] < x < event1[0]:
                                        relative_distance = abs(x - event1[0])
                                        basepair_difference = relative_distance * extension
                                        if x > event1[0]:
                                            measured_position = event1[1] - basepair_difference
                                            calc_error = measured_position - bptarget
                                            self.points_pairwise.append((x, bptarget, measured_position,
                                                                         relative_distance, calc_error,
                                                                         twoevents, barcode))
                                        else:
                                            measured_position = event1[1] + basepair_difference
                                            calc_error = measured_position - bptarget
                                            self.points_pairwise.append((x, bptarget, measured_position,
                                                                         relative_distance, calc_error,
                                                                         twoevents, barcode))
                                    else:
                                        pass
                        else:
                            pass

    def pair_check(self, excluded1, excluded2):
        if len(self.points_targets) > 2:
            pairs = [each for each in combinations(self.points_targets, 2)]
            for twoevents in pairs:
                event1 = twoevents[0]
                event2 = twoevents[1]
                if event1[1] == event2[1]:
                    pass
                elif event1[1] == excluded1 or event2[1] == excluded1:
                    pass
                elif event1[1] == excluded2 or event2[1] == excluded2:
                    pass
                else:
                    self.pairs_used.append(twoevents)

    def pairs_list(self):
        if (len(self.pairs_used)):
            return [i for i in self.pairs_used]
        else:
            return None

    def strand_parameters(self):
        if (len(self.strand_extension)):
            return [i for i in self.strand_extension]
        else:
            return None

    def calculated_position(self):
        if(len(self.points_pairwise)):
            return[i for i in self.points_pairwise]
        else:
            return None

    def pairwise(self):
        """

        :return:
        """
        if len(self.scaled_points) > 1:
            arr = np.array(self.scaled_points)
            unique = np.unique(distance.cdist(arr[:, :1], arr[:, :1],
                                              'euclidean'))
            return unique.tolist()
        elif len(self.assigned_points) > 1:
            print("WARNING: Pairwise calculated from unscaled data")
            arr = np.array(self.assigned_points)
            unique = np.unique(distance.cdist(arr[:, :2], arr[:, :2],
                                              'euclidean'))
            return unique.tolist()
