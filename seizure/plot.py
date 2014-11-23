'''
Created on Oct 18, 2014

@author: newuser
'''
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy

def plot(interictal, preictal, axis1, axis2):
    x1 = numpy.array([i[axis1] for i in interictal])
    x2 = numpy.array([i[axis1] for i in preictal])
    y1 = numpy.array([i[axis2] for i in interictal])
    y2 = numpy.array([i[axis2] for i in preictal])

    # definitions for the axes
    left, width = 0.1, 0.8
    bottom, height = 0.1, 0.8

    rect_scatter = [left, bottom, width, height]

    # start with a rectangular Figure
    plt.figure(1, figsize=(8,8))

    axScatter = plt.axes(rect_scatter)

    # the scatter plot:
    axScatter.scatter(x1, y1, color="r")
    axScatter.scatter(x2, y2, color="b")

    plt.show()