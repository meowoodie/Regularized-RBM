#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module contains basic interface used throught the whole holmes package.

The interfaces are realized as abstract base classes for the basic natural
language processing.
"""

from gensim import corpora, models
from nltk.util import ngrams
from six import iteritems
import numpy as np
import string
import arrow
import nltk
import sys
import re

reload(sys)
sys.setdefaultencoding('utf8')

class Documents(object):
	"""
	Documents is a simple class for:
	1. preprocessing documents text in a memory-friendly way.
	2. outputting tokenized terms of each of the documents iteratively.

	It's mainly used to load the documents and help substantiate gensim's
	dictionary and corpus.

	Essentially, a documents object is simply an iterable, where each iteration
	step yields one document, then splits and formats their words by an unified
	and standard method.
	"""

	def __init__(self, iter_object, n=1, pad_right=False, pad_left=False, \
		         left_pad_symbol=None, right_pad_symbol=None, keep_sents=False):
		self.iter_object = iter_object
		self.counter     = 0
		self.n           = n
		self.pad_right   = pad_right
		self.pad_left    = pad_left
		self.left_pad_symbol  = left_pad_symbol
		self.right_pad_symbol = right_pad_symbol
		self.keep_sents  = keep_sents

	def __iter__(self):
		"""
        Iterate over the corpus, yielding one line (document) at a time.
        """
		for line in self.iter_object:
			if self.counter > 0 and self.counter % 1000 == 0:
				print >> sys.stderr, "[%s] [Documents] %s docs have been processed." % \
				         (arrow.now(), self.counter)
			try:
				yield self.tokenize(line, N=self.n, \
                                    pad_right=self.pad_right, \
                                    pad_left=self.pad_left, \
					                left_pad_symbol=self.left_pad_symbol, \
					                right_pad_symbol=self.right_pad_symbol, \
									keep_sents=self.keep_sents)
			# Yield empty token list if tokenization failed as UnicodeDecodeError was raised
			except UnicodeDecodeError as e:
				print >> sys.stderr, "[%s] [Documents] No. %s doc raise expection: %s." % \
				         (arrow.now(), self.counter, e)
				yield []

			self.counter += 1

	@staticmethod
	def tokenize(text_string, N=1, pad_right=False, pad_left=False, \
		         left_pad_symbol=None, right_pad_symbol=None, keep_sents=False):
		"""
		Tokenize each of the words in the text (one document).

		It utilizes nltk to help tokenize the sentences and the words in the
		text. What needs to be noted is one document is consist of multiple
		remarks, which are delimited by "/1" within the text.

		Also, parameters of ngrams module, like n, pad_right, pad_left,
		left_pad_symbol, and right_pad_symbol, are optional to input.

		Note:
		If the yield documents are going to be used to generate vocabulary,
		'keep_sents' has to be set False, since the creation of dictionary only
		take 1D documents as input.
		"""
		ngram_tokens = []
		# Free text part for each of the records are delimited by "\1"
		for remark in text_string.strip().split("\1"):
			# For every sentences in each of the free text part
			for sent in nltk.tokenize.sent_tokenize(remark.encode('utf-8').strip()):
				# Tokenize a sentence by english word level of granularity
				tokens_in_sentence = [
					token
					for token in nltk.word_tokenize(sent.translate(None, string.punctuation).lower())
					if token not in nltk.corpus.stopwords.words("english")]
				# Calculate all the grams terms from unigram to N-grams
				for n in range(1, N+1):
					# Calculate ngram of a tokenized sentence
					ngram_tokens_in_sentence = [
						"_".join(ngram_tuple)
						for ngram_tuple in \
							list(ngrams(tokens_in_sentence, n, pad_right=pad_right, pad_left=pad_left, \
								        left_pad_symbol=left_pad_symbol, \
								        right_pad_symbol=right_pad_symbol)) ]
					# Append ngrams terms to the list
					if keep_sents:
						ngram_tokens.append(ngram_tokens_in_sentence)
					else:
						ngram_tokens += ngram_tokens_in_sentence
		return ngram_tokens



def dictionary(text_iter_obj, min_term_freq=1, \
               n=1, pad_right=False, pad_left=False, \
			   left_pad_symbol=None, right_pad_symbol=None):
	"""
	"""
	# Init document object by loading an iterable object (for reading text iteratively),
	# the iterable object could be a file handler, or standard input handler and so on
	docs = Documents(text_iter_obj, n=n, pad_right=pad_right, pad_left=pad_left,
					 left_pad_symbol=left_pad_symbol,
					 right_pad_symbol=right_pad_symbol)
	# Build dictionary based on the words appeared in documents
	dictionary = corpora.Dictionary(docs)
	# Remove non-character and low-frequency terms in the dictionary
	nonchar_ids = [ tokenid \
	                for token, tokenid in iteritems(dictionary.token2id) \
	                if not re.match("^[A-Za-z_]*$", token) ]
	lowfreq_ids = [ tokenid
	                for tokenid, docfreq in iteritems(dictionary.dfs) \
	                if docfreq <= min_term_freq ]
	dictionary.filter_tokens(lowfreq_ids + nonchar_ids)
	# Remove gaps in id sequence after some of the words being removed
	dictionary.compactify()
	return dictionary



def corpus_by_sentences(text_iter_obj, dictionary, \
                        n=1, pad_right=False, pad_left=False, \
						left_pad_symbol=None, right_pad_symbol=None):
	"""
	Build

	It would process the documents in the raw text file interatively by handing
	a iterable object "text_iter_obj". It requires each line of the raw text
	file only contains a single document. During the mean time, the function
	would generate a dictionary file which contains all the non-stop english
	words (vocabulary) appeared in the corpus at least "min_term_freq" times.
	It contributes to less storage space for corpus and easier/faster corpus
	operations.
	"""
	docs = Documents(text_iter_obj, n=n, pad_right=pad_right, pad_left=pad_left,
					 left_pad_symbol=left_pad_symbol,
					 right_pad_symbol=right_pad_symbol,
					 keep_sents=True)
	# Build corpus (numeralize the documents and only keep the terms that exist in dictionary)
	# corpus = [ [ dictionary.doc2bow(sent) for sent in doc ] for doc in docs ]
	corpus = []
	for doc in docs:
		for sent in doc:
			corpus.append(dictionary.doc2bow(sent))
	# Calculate tfidf matrix
	tfidf = models.TfidfModel(corpus)
	tfidf_corpus = tfidf[corpus]
	return tfidf_corpus
