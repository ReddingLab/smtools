#!/usr/bin/env python

"""
The misc module contains commonly used functions
"""

__all__ = ['skewnormal','super_gauss_function','mundane_gauss_function']

__version__ = '0.2.0'

__author__ = 'Sy Redding'


import numpy as np
from scipy.special import erf

def skewnormal(x, loc, scale, shape, amplitude, baseline):
    """
    `skewnormal distribution <https://en.wikipedia.org/wiki/Skew_normal_distribution/>`

    :param x: variable
    :param loc: location parameter, mean
    :param scale: scale parameter, variance
    :param shape: shape parameter, skew
    :param amplitude:
    :param baseline:
    :return:
    """
    t = (x - loc) / scale
    pdf = 1/np.sqrt(2*np.pi) * np.exp(-t**2/2)
    cdf = (1 + erf((t*shape)/np.sqrt(2))) / 2.
    return 2*amplitude / scale * pdf * cdf + baseline


def super_gauss_function(x, baseline, amplitude, mean, sigma, power):
    """
    `super Guassian distribution`

    :param x:
    :param baseline:
    :param amplitude:
    :param mean:
    :param sigma:
    :param power:
    :return:
    """
    return baseline + amplitude * np.exp(
        (-((x - mean) ** 2 / (2 * sigma ** 2)) ** power))


def mundane_gauss_function(x, baseline, amplitude, mean, sigma):
    """
    `regular ol' Guassian distribution`

    :param x:
    :param baseline:
    :param amplitude:
    :param mean:
    :param sigma:
    :return:
    """
    return baseline + amplitude * np.exp(
        (-((x - mean) ** 2 / (2 * sigma ** 2))))