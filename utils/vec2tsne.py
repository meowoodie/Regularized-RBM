#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script takes numpy text file, which is a 2D matrix, as input, and calculate
t-sne embeddings for each row vector in the matrix and plot the embedding vectors
at a 2D space.
"""
import sys
import argparse
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

np.random.seed(100)

# def sents_loader(label_path, delimiter="\t"):
# 	"""
# 	Sentences Loader
#
# 	Load sentences information from text file. Including:
# 	- index of document in corpus
# 	- index of sentence in document
# 	- number of sentences in document
# 	- text string of sentence
# 	"""
# 	labels      = []
# 	annotations = []
# 	with open(label_path, "r") as fhandler:
# 		for line in fhandler:
# 			sent, sent_ind, sent_num, doc_ind = line.strip().split(delimiter)
# 			# Strings of sentences.
# 			annotations.append("[%s] (%s in %s) %s" % (doc_ind, sent_ind, sent_num, sent))
# 			# Labels ranges from int 0 to 9.
# 			labels.append(int((float(sent_ind)/float(sent_num)) * 10))
# 	return labels, annotations

# def sents_info_loader(label_path, delimiter="\t"):
# 	"""
# 	"""
# 	labels      = []
# 	annotations = []
# 	with open(label_path, "r") as fhandler:
# 		for line in fhandler:
# 			sent, sent_ind, sent_num, doc_ind, catagory, time, lat, lon = line.strip().split(delimiter)
# 			# Strings of sentences.
# 			annotations.append("[%s] <%s> (%s in %s) %s" % (doc_ind, catagory, sent_ind, sent_num, sent))
# 			# Labels ranges from int 0 to 9.
# 			labels.append(catagory)
# 			# if lat == "none":
# 			# 	labels.append(84.42156)
# 			# else:
# 			# 	labels.append(float(lon)/100000)
# 	labels_set = list(set(labels))
# 	colorbar = ['g', 'y', 'r', 'c', 'm', 'b', 'k', 'w']
# 	labels = [ colorbar[labels_set.index(label)] for label in labels ]
# 	# label_range  = max(labels) - min(labels)
# 	# labels = [ (label - min(labels))/label_range * 10 for label in labels ]
# 	return labels, annotations

annotations = []
def docs_info_loader(label_path, delimiter="\t"):
	"""
	"""
	labels      = []
	with open(label_path, "r") as fhandler:
		for line in fhandler:
			doc_ind  = line.strip().split(delimiter)[0]
			catagory = line.strip().split(delimiter)[1]
			labels.append(catagory)
			annotations.append("[%s] %s" % (doc_ind, catagory))

	# labels_set   = list(set(labels))
	color_labels = [
		"burglary", "pedrobbery", "dijawan_adams",
		"jaydarious_morrison", "julian_tucker", "thaddeus_todd"]
	colorbar = ["g", "r", "c", "m", "b", "k"]

	# debug
	# print len(labels_set)

	colors = []
	for label in labels:
		if label in color_labels:
			colors.append(colorbar[color_labels.index(label)])
		else:
			colors.append("y")
	# colors = [ labels_set.index(label) for label in labels ]

	return colors, annotations

def update_annot(ind):
	pos = sc.get_offsets()[ind["ind"][0]]
	annot.xy = pos
	text = "\n".join([annotations[i] for i in ind["ind"]])
	annot.set_text(text)

def hover(event):
	vis = annot.get_visible()
	if event.inaxes == ax:
		cont, ind = sc.contains(event)
		if cont:
			update_annot(ind)
			annot.set_visible(True)
			fig.canvas.draw_idle()
		else:
			if vis:
				annot.set_visible(False)
				fig.canvas.draw_idle()

# Parse the input parameters
parser = argparse.ArgumentParser(description="Script for converting vectors to t-SNE projections")
parser.add_argument("-v", "--vpath", required=True, help="The path of the numpy txt file")
parser.add_argument("-l", "--lpath", required=True, help="The path of the label txt file")
# Input parameters
args = parser.parse_args()
vec_path = args.vpath
lab_path = args.lpath
tsne_dim = 2
# Load labels and annotations
labels, annotations = docs_info_loader(lab_path)
# Load or calculate embedded vectors
vectors = np.loadtxt(vec_path, delimiter=",")
print >> sys.stderr, "input vectors with size (%d, %d) have been loaded" % vectors.shape
embedded_vecs = TSNE(n_components=tsne_dim).fit_transform(vectors)
print >> sys.stderr, "embeddings vectors with size (%d, %d) have been generated" % embedded_vecs.shape

# np.savetxt("resource/embeddings/events/tsne-spatial-trigram-tfidf-vecs.txt", embedded_vecs, delimiter=',')
# embedded_vecs = np.loadtxt(vec_path, delimiter=",")

# Init plot and colorbar
fig,ax = plt.subplots()
cm = plt.cm.get_cmap('RdYlBu')

# Plot scatter points
embedded_vecs = np.flip(embedded_vecs, 0) # to make labeled cases on the top of other points
labels        = np.flip(labels, 0)
annotations   = np.flip(annotations, 0)
sc = plt.scatter(embedded_vecs[:, 0], embedded_vecs[:, 1], c=labels, s=20)
# sc = plt.scatter(embedded_vecs[:, 0], embedded_vecs[:, 1], c=labels, vmin=0, vmax=len(list(set(labels))), s=20, cmap=cm)

# Set initial annotations for plot
annot = ax.annotate("", xy=(0,0), xytext=(1,1), textcoords="offset points")
annot.set_visible(False)
# Set trigger event (hover annotations) for mouse
fig.canvas.mpl_connect("motion_notify_event", hover)

# Format plot
plt.axis('off')
plt.tight_layout()
plt.show()
