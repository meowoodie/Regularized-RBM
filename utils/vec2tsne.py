#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script takes numpy text file, which is a 2D matrix, as input, and calculate
t-sne embeddings for each row vector in the matrix and plot the embedding vectors
at a 2D space.
"""
from __future__ import print_function

from sklearn.manifold import TSNE
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import numpy as np
import argparse
import random
import sys

np.random.seed(100)

LABELS = [ "burglary", "pedrobbery", "dijawan_adams", \
           "jaydarious_morrison", "julian_tucker", "thaddeus_todd"]
TEXTS  = [ "Burglary in Buckhead", "Ped Robbery in Buckhead", \
           "Ped Robbery by Suspect A", "Ped Robbery by Suspect M", \
		   "Ped Robbery by Suspect J", "Ped Robbery by Suspect T"]
COLORS = [ "g", "r", "c", "m", "b", "k" ]

def label_loader(label_path, delimiter="\t"):
	"""
	Load labeling information from local file. If label does not appear in
	LABELS then it will be assigned as "random".
	"""
	labels      = []
	annotations = []
	with open(label_path, "r") as fhandler:
		for line in fhandler:
			doc_ind  = line.strip().split(delimiter)[0]
			catagory = line.strip().split(delimiter)[1]
			labels.append(catagory)
			annotations.append("[%s] %s" % (doc_ind, catagory))

	# assign "random" if label is not in LABELS
	labels = [ x if x in LABELS else "random" for x in labels ]

	return labels, annotations

def vec2tsne(lab_path, plot_path, vectors=None, vec_path=None, n=2):
	"""
	Function for projecting high-dimensional vecters to a n-dimensional t-SNE space and
	color the points with labeling information.
	"""
	if vec_path is not None:
		# loading vectors from a numpy file
		vectors = np.loadtxt(vec_path, delimiter=",")
		print("input vectors with size (%d, %d) have been loaded" % \
		      vectors.shape, file=sys.stderr)
	if vectors is None:
		raise Exception("Please indicate your vectors")
	# project vectors to a n-D t-SNE space
	embedded_vecs = TSNE(n_components=n).fit_transform(vectors)
	print("embeddings vectors with size (%d, %d) have been generated" % \
	      embedded_vecs.shape, file=sys.stderr)
	# load labels and corresponding annotations
	labels, annotations = label_loader(lab_path)

	# organize the points in a proper order (reverse) in avoid of random points
	# coverring labeled points in the plot
	embedded_vecs = np.flip(embedded_vecs, 0) # to make labeled cases on the top of other points
	labels        = np.flip(labels[0:256], 0)
	annotations   = np.flip(annotations[0:256], 0)

	# plot as a pdf file
	with PdfPages(plot_path) as pdf:
		fig, ax = plt.subplots(1, 1)
		for label in list(set(labels)):
			indices = [ i for i, x in enumerate(labels) if x == label ]
			color   = COLORS[LABELS.index(label)] if label in LABELS else "y"
			text    = TEXTS[LABELS.index(label)] if label in LABELS else "Random Case"
			plt.scatter(embedded_vecs[indices, 0], embedded_vecs[indices, 1], c=color, \
			            label=text, edgecolors='none', s=20)
		plt.legend()
		plt.axis('off')
		plt.tight_layout()
		pdf.savefig(fig)

if __name__ == "__main__":

	# Parse the input parameters
	parser = argparse.ArgumentParser(description="Script for converting vectors to t-SNE projections")
	parser.add_argument("-v", "--vpath", required=True, help="The path of the numpy txt file")
	parser.add_argument("-l", "--lpath", required=True, help="The path of the label txt file")
	parser.add_argument("-p", "--ppath", required=True, help="The path of the plot result")
	# Input parameters
	args = parser.parse_args()
	vec_path = args.vpath
	lab_path = args.lpath
	plt_path = args.ppath

	vec2tsne(lab_path, vec_path=vec_path, plot_path="results/test.pdf")
