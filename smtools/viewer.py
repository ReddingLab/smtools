#!/usr/bin/env python

"""
The viewer module contains a handy image viewer for stacks
"""

__all__ = ['StackViewer']

__version__ = '0.3.0'

__author__ = 'Sy Redding'


import matplotlib.pyplot as plt
import numpy as np


class StackViewer(object):
    def __init__(self, ax, image_stack, overlays=None):

        self.ax = ax
        self.stack = image_stack
        self.over_stacks = overlays

        if self.over_stacks is not None:
            if self.stack.shape[0] != len(self.over_stacks):
                print("EXITING..."
                      "overlay stacks must be same length as "
                      "parent stack")
                exit()

        self.ind = 0

        self.RGB = False
        self.gray = False
        self.vmax = 1

        if len(self.stack.shape) == 4:
            self.RGB = True
            self.slices, rows, cols, colors = self.stack.shape
            self.im = ax.imshow(self.stack[self.ind, :, :, :],
                                vmax=self.vmax)

        elif len(self.stack.shape) == 3:
            self.gray = True
            self.slices, rows, cols = self.stack.shape
            self.im = ax.imshow(self.stack[self.ind, :, :],
                                cmap="Greys_r", vmax=self.vmax)

        marker_style = dict(color='orangered', linestyle='none',
                            marker='o', markersize=5, fillStyle='none')
        if self.over_stacks is not None:
            self.points, = self.ax.plot([], [], **marker_style)

        ax.set_position((.05, .3, .9, .6))
        string = (
            "Scroll through images or use $\u2190$ and $\u2192$ \n"
            "Use $\u2191$ and $\u2193$ to change the contrast \n")
        ax.text(cols / 2, rows + (rows / 3), string, ha='center',
                fontsize=14, fontweight='bold', wrap=True)

        self.update()

    def onpress(self, event):
        if event.key == 'right':
            self.ind = (self.ind + 1) % self.slices
        elif event.key == 'left':
            self.ind = (self.ind - 1) % self.slices

        if event.key == 'up':
            if self.vmax < 1:
                self.vmax = self.vmax + .03

        elif event.key == 'down':
            if self.vmax > .03:
                self.vmax = self.vmax - .03
        self.update()

    def onscroll(self, event):
        if event.button == 'down':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        if self.RGB:
            self.im.set_data(self.stack[self.ind, :, :, :])
        elif self.gray:
            self.im.set_data(self.stack[self.ind, :, :])
        if self.over_stacks is not None:
            xdata, ydata = map(list, zip(*self.over_stacks[self.ind]))
            self.points.set_data(xdata, ydata)
        self.ax.set_ylabel('slice %s' % self.ind, fontsize=14,
                           fontweight='bold')
        self.im.axes.figure.canvas.draw()
        self.im.set_clim(vmax=self.vmax)


def mkViewer(imagestack, overstack=None):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    imagestack = np.stack(imagestack)
    view = StackViewer(ax, imagestack, overstack)
    fig.canvas.mpl_connect('key_press_event', view.onpress)
    fig.canvas.mpl_connect('scroll_event', view.onscroll)
    plt.show()

