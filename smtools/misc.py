#!/usr/bin/env python

"""
The misc module contains commonly used functions.

"""

__all__ = ['two_d_gaussian', 'skewnormal', 'super_gauss_function',
           'mundane_gauss_function', 'straight_line']

__version__ = '0.3.0'

__author__ = 'Sy Redding'

import numpy as np
from scipy.special import erf


# -------------------------------------------------------------------- #
# ----------------------- Gaussian functions ------------------------- #
# -------------------------------------------------------------------- #


def two_d_gaussian(span, amplitude, mu_x, mu_y, sigma_x, sigma_y, theta,
                   baseline):
    """

    `2D Gaussian distribution
    <https://en.wikipedia.org/wiki/Gaussian_function#Two-dimensional_Gaussian_function>`_

    .. math::
        :nowrap:

        \\begin{gather*}
        f(x,y) = \\text{baseline} + A \\exp \\left [ -(x-\\mu_x)^2 -
        2b (
        x-\\mu_x)(
        y-\\mu_y) - c(y-\\mu_y)^2 \\right ]  \\\\

        a = \\frac{\\cos^2\\theta}{2\\sigma_x^2} +
            \\frac{\\sin^2\\theta}{2\\sigma_y^2}  \\\\

        b = -\\frac{\\sin 2\\theta}{4\\sigma_x^2} +
            \\frac{\\sin 2\\theta}{4\\sigma_y^2} \\\\

        a = \\frac{\\sin^2\\theta}{2\\sigma_x^2} +
            \\frac{\\cos^2\\theta}{2\\sigma_y^2}
        \\end{gather*}

    """
    (x, y) = span
    mu_x = float(mu_x)
    mu_y = float(mu_y)
    a = (np.cos(theta) ** 2) / (2 * sigma_x ** 2) + (
            np.sin(theta) ** 2) / (2 * sigma_y ** 2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x ** 2) + (
        np.sin(2 * theta)) / (4 * sigma_y ** 2)
    c = (np.sin(theta) ** 2) / (2 * sigma_x ** 2) + (
            np.cos(theta) ** 2) / (2 * sigma_y ** 2)
    g = baseline + amplitude * np.exp(
        - (a * ((x - mu_x) ** 2) + 2 * b * (x - mu_x) * (y - mu_y)
           + c * ((y - mu_y) ** 2)))
    return g.ravel()


def super_gauss_function(x, baseline, amplitude, mean, sigma, power):
    """

    `Higher order Gaussian
    <https://en.wikipedia.org/wiki/Gaussian_function#Higher-order_Gaussian_or_super-Gaussian_function>`_

    .. math::

        f(x) = \\text{baseline} +
                A \\exp \\left [ \\left ( \\frac{x-\\mu_x}{
                2\\sigma^2_x}\\right)^p\\right]
    """
    return baseline + amplitude * np.exp(
        (-((x - mean) ** 2 / (2 * sigma ** 2)) ** power))


def mundane_gauss_function(x, baseline, amplitude, mean, sigma):
    """

    `Gaussian distribution
    <https://en.wikipedia.org/wiki/Gaussian_function>`_


    .. math::

        f(x) = \\text{baseline} +
                A \\exp \\left [ \\frac{x-\\mu_x}{
                2\\sigma^2_x}\\right]
    """
    return baseline + amplitude * np.exp(
        (-((x - mean) ** 2 / (2 * sigma ** 2))))


def skewnormal(x, loc, scale, shape, amplitude, baseline):
    """

    `skewnormal distribution
    <https://en.wikipedia.org/wiki/Skew_normal_distribution>`_

    .. math::
        :nowrap:

        \\begin{gather*}
        f(x) = \\text{baseline} + \\frac{2A}{\\text{scale}} \\phi(t)
        \\Phi(\\alpha t)  \\\\

        \\phi(x) = \\frac{1}{\\sqrt{2 \\pi}} e^{- \\frac{t^2}{2}} \\\\

        \\Phi(x) = \\frac{1}{2} \\left [1+ \\text{erf} \\left(
                   \\frac{t}{ \\sqrt{2}}  \\right) \\right ]  \\\\

        t \\to \\frac{x-\\text{location}}{\\text{scale}}

        \\end{gather*}
    """
    t = (x - loc) / scale
    pdf = 1 / np.sqrt(2 * np.pi) * np.exp(-t ** 2 / 2)
    cdf = (1 + erf((t * shape) / np.sqrt(2))) / 2.
    return 2 * amplitude / scale * pdf * cdf + baseline


# -------------------------------------------------------------------- #
# ---------------------- Exponential functions ----------------------- #
# -------------------------------------------------------------------- #

# -------------------------------------------------------------------- #
# ------------------------ Linear functions -------------------------- #
# -------------------------------------------------------------------- #

def straight_line(x, m, b):
    """

    `line
    <https://en.wikipedia.org/wiki/Line_(geometry)#On_the_Cartesian_plane>`_

    .. math::

        f(x) = mx+b
    """
    return m * x + b
