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
from gensim import corpora

def matrix_plotter(matrix, path="results/corpus_embeddings.pdf"):
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    with PdfPages(path) as pdf:
        fig, ax = plt.subplots(1, 1)
        im  = ax.imshow(matrix, cmap=cm.Greys_r, interpolation='nearest')
        ax.set_ylabel("documents")
        ax.set_xlabel("keywords")
        ax.set_aspect('auto')
        fig.colorbar(im)
        pdf.savefig(fig)

def error_plotter(errs, labels=[], path="results/errors.pdf"):
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    # plot as a pdf file
    with PdfPages(path) as pdf:
        fig, ax = plt.subplots(1, 1)
        for ind in range(len(labels)):
            plt.plot(errs[ind], label=labels[ind], linewidth=4.0)
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
            plt.plot(zros[ind], label=labels[ind], linewidth=4.0)
        plt.legend()
        plt.xlabel("Iteration")
        plt.ylabel("Number of Eliminated Variables")
        plt.grid(True)
        plt.tight_layout()
        pdf.savefig(fig)

def cv_plotter(errs, zros, lam, path="results/cv.pdf"):
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    # performance formula
    err = errs.mean(axis=1)
    zro = zros.mean(axis=1)
    perf = err / np.exp((zro-zro.min()+1)/500)
    perf_max = errs.max(axis=1) / np.exp((zros.min(axis=1) - zros.min())/500)
    perf_min = errs.min(axis=1) / np.exp((zros.max(axis=1) - zros.min())/500)
    # plot as a pdf file
    with PdfPages(path) as pdf:
        fig, ax = plt.subplots(1, 1)
        plt.plot(lam, perf, c="blue", linestyle=":", linewidth=2.0)
        ax.errorbar(lam, perf, yerr=[perf_min, perf_max], \
                    fmt='*', ecolor="gray", capthick=1)
        plt.xlabel("$\log \lambda$")
        plt.ylabel("Error / $\exp$(Number of zero variables)")
        # plt.grid(True)
        plt.tight_layout()
        pdf.savefig(fig)

def intensity_plotter(intensity, dictionary, k=20, path="results/intensity.pdf"):
    # top k keywords according to intensity
    top_k_inds = intensity.argsort()[-1*k:][::-1]
    top_k_keywords = [ dictionary[ind] for ind in top_k_inds ]
    # print(top_k_keywords)
    # plot as a pdf file
    with PdfPages(path) as pdf:
        fig, ax = plt.subplots(1, 1, figsize=(10, 3))
        ax.vlines(range(len(intensity)), [0], intensity)
        ax.plot(top_k_inds, intensity[top_k_inds], '^')
        for i, keyword in enumerate(top_k_keywords):
            ax.annotate(keyword, (top_k_inds[i], intensity[top_k_inds][i]+1e-4), size=5, color="red")
        ax.set_ylabel('Standard deviation of tf-idf intensity')
        ax.set_xlabel('Keywords')
        # Turn off tick labels
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        plt.tight_layout()
        pdf.savefig(fig)

def bar_plotter(path="results/fscore.pdf"):
    # dataset
    # RBM with regularization
    score_reg_rbm = [0.15831824908807041, 0.14106623564416126, 0.1094700518782232, 0.096323801425203318]
    # regular RBM
    score_rbm = [0.076563426203208898, 0.072932631783345955, 0.063187276466536366, 0.055808217241579479]
    # SVD
    score_svd = [0.15337031347221972, 0.13769841598939312, 0.11522019368164071, 0.10800105552200189]
    # LDA
    score_lda = [0.054438022759614353, 0.065424181538917603, 0.0711249791059681, 0.069326223352732738]
    # Autoencoder
    score_ae  = [0.020288035440106472, 0.014952014571509634, 0.012322640429558076, 0.012212399671753981]
    # plot as pdf
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    with PdfPages(path) as pdf:
        N = 4
        ind = np.arange(4) # the x locations for the groups
        width = 0.1        # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(ind - 2*width, score_reg_rbm, width, color='red')
        rects2 = ax.bar(ind - width, score_rbm, width, color='orange')
        rects3 = ax.bar(ind, score_svd, width, color='blue')
        rects4 = ax.bar(ind + width, score_lda, width, color='green')
        rects5 = ax.bar(ind + 2*width, score_ae, width, color='grey')

        # add some text for labels, title and axes ticks
        ax.set_ylabel('$F_1$ score')
        ax.set_xlabel('Number of selected events given a query ($N$)')
        ax.set_xticks(ind)
        ax.set_xticklabels(('20', '40', '80', '100'))

        ax.legend((rects1[0], rects2[0], rects3[0], rects4[0], rects5[0]),
            ('regularized RBM ($\lambda=10^{-3}$)', 'RBM ($\lambda=0$)', 'Truncated SVD', 'LDA', 'Denoising Autoencoder'))

        plt.grid(True)
        plt.tight_layout()
        pdf.savefig(fig)

if __name__ == "__main__":
    # PLOT ERROS AND ZEROS RESPECTIVELY
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

    # PLOT CROSS-VALIDATION FOR LAMBDA
    errs = np.loadtxt("resource/cv_errs.txt", delimiter=",")
    zros = np.loadtxt("resource/cv_zeros.txt", delimiter=",")
    cv_plotter(errs, zros, np.linspace(-15, 0, num=31)[1:], path="results/cv.pdf")

    # PLOT EMBEDDINGS
    # dictionary
    dict_name  = "resource/dict/2k.bigram.dict"
    ngram_dict = corpora.Dictionary.load(dict_name)
    # load embeddings
    embeddings = np.loadtxt("resource/embeddings/reg.1e-3.lr.1e-3.2k.recon.txt", delimiter=",")
    # convert to intensity
    embeddings[embeddings < 1e-2] = 1e-5
    intensity = (embeddings).std(axis=0)
    intensity_plotter(intensity, ngram_dict)

    # PLOT F-MEASURE BAR FOR EMBEDDINGS
    bar_plotter()
