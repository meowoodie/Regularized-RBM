#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script provides functions for visualizing 1D & 2D tensors (array and matrix).
"""

import sys
import argparse
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def matrix_plotter(matrix):
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    im  = ax.imshow(matrix, cmap=cm.Greys_r) # , interpolation='nearest'
    fig.colorbar(im)
    plt.show()

def error_plotter(errs, labels=[], path="results/errors.pdf"):
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    # plot as a pdf file
    with PdfPages(path) as pdf:
        fig, ax = plt.subplots(1, 1)
        for ind in range(len(labels)):
            plt.plot(errs[ind], label=labels[ind])
        plt.legend()
        plt.xlabel("Iteration")
        plt.ylabel("Mean Square Error")
        plt.grid(True)
        plt.tight_layout()
        pdf.savefig(fig)

def zeros_plotter(zros, labels=[], path="results/zeros.pdf"):
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    # plot as a pdf file
    with PdfPages(path) as pdf:
        fig, ax = plt.subplots(1, 1)
        for ind in range(len(labels)):
            plt.plot(zros[ind], label=labels[ind])
        plt.legend()
        plt.xlabel("Iteration")
        plt.ylabel("Number of Eliminated Variables")
        plt.grid(True)
        plt.tight_layout()
        pdf.savefig(fig)

if __name__ == "__main__":
    # plot errors
    err1 = np.loadtxt("resource/errors.lam1e-03.lr1e-03.t1e-02.epoch20.txt", delimiter=",")
    err2 = np.loadtxt("resource/errors.lam5e-04.lr1e-03.t1e-02.epoch20.txt", delimiter=",")
    err3 = np.loadtxt("resource/errors.lam1e-04.lr1e-03.t1e-02.epoch20.txt", delimiter=",")
    err4 = np.loadtxt("resource/errors.lam0e+00.lr1e-03.t1e-02.epoch20.txt", delimiter=",")
    errs = np.stack([err1, err2, err3, err4], axis=1).transpose()

    error_plotter(errs, labels=[
        r'$\lambda=1 \times 10^{-3}$',
        r'$\lambda=5 \times 10^{-4}$',
        r'$\lambda=1 \times 10^{-4}$',
        r'$\lambda=0$ (without penalty)'],
        path="results/errors.pdf")

    # plot zeros
    zro1 = np.loadtxt("resource/zeros.lam1e-03.lr1e-03.t1e-02.epoch20.txt", delimiter=",")
    zro2 = np.loadtxt("resource/zeros.lam5e-04.lr1e-03.t1e-02.epoch20.txt", delimiter=",")
    zro3 = np.loadtxt("resource/zeros.lam1e-04.lr1e-03.t1e-02.epoch20.txt", delimiter=",")
    zro4 = np.loadtxt("resource/zeros.lam0e+00.lr1e-03.t1e-02.epoch20.txt", delimiter=",")
    zros = np.stack([zro1, zro2, zro3, zro4], axis=1).transpose()

    zeros_plotter(zros, labels=[
        r'$\lambda=1 \times 10^{-3}$',
        r'$\lambda=5 \times 10^{-4}$',
        r'$\lambda=1 \times 10^{-4}$',
        r'$\lambda=0$ (without penalty)'],
        path="results/zeros.pdf")
