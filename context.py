#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
"""
import arrow
import numpy as np
from scipy.spatial.distance import pdist, squareform

class Context(object):
    """
    context

    It's a utility class for handling rawdata and generating context training
    dataset for (modified) skipgram model. The context information mainly
    includes temporal context, spatial context, catagory context and so on.

    As an example in text, "the quick brown fox jumped over the lazy dog" have
    (context, target) pairs like: "([the, brown], quick), ([quick, fox], brown),
    ([brown, jumped], fox), ..."
    """

    def __init__(self, file_name):
        self.rawdata = []
        with open(file_name, "r") as fhandler:
            ind = 0
            for line in fhandler:
                _id, catagory, time, lat, lon = line.strip().split("\t")
                #TODO: clean data and simplify the procedure of time preprocessing
                time = time.split("T")[0]
                year, month, day = [ int(t) for t in time.split("-") ]
                time = int(arrow.get(year, month, day).timestamp)
                # Get latitude and longitude of event
                lat  = float(lat)/100000.0
                lon  = -1 * float(lon)/100000.0
                self.rawdata.append([ind, _id, catagory, time, lat, lon])
                ind += 1

    def spatial_context(self, delta=0.05):
        """
        Spatial context

        Getting spatial context for each of the event in rawdata. Spatial
        context is defined by the vicinity of the event in the Euclidean space.
        """

        # def is_context(target, context, max_dist):
        #     delta = distance(target, context)
        #     if max_dist > delta:
        #         return True
        #     else:
        #         return False

        gps_points = np.array([ row[4:6] for row in self.rawdata ])
        # Calculate pairwise distance between gps points
        dists = pdist(gps_points, "euclidean")
        # Convert distances into squareforms for easier retrival
        dists = squareform(dists)
        # Return context for each of points
        indices = range(len(gps_points))
        context = [ (row_ind, [ col_ind
                                for (col_ind, dist) in zip(indices, dists[row_ind])
                                if dist <= delta ])
                    for row_ind in indices ]
        return context


    def temporal_context(self, delta=0.05):
        """
        Temporal Context

        Getting temporal context for each of the event in rawdata. Temporal
        context is defined by the vicinity of the event over the time.
        """
        pass

    @staticmethod
    def target_context_pairs(context):
        """
        Formatting context list to (target, context) pairs.
        E.g.
        context list: ("ti" is target, "cij" is context of "ti")
            [(t1, [c11, c12, ...]),
             (t2, [c21, c22, ...]),
             ...
             (tm, [cm1, cm2, ...])]
        pairs list:
            [[t1, [c11]], [t1, [c12]], ... [t2, [c21]], ..., [tm, [cm1]], ...]
        """
        tis  = []
        cijs = []
        for ti, ci in context:
            for cij in ci:
                tis.append(ti)
                cijs.append([cij])
        return tis, cijs

if __name__ == "__main__":
    # Unittest for Context
    c  = Context("data/info.txt")
    sc = c.spatial_context(delta=0.05)
    input_data, label_data = c.target_context_pairs(sc)
    print len(input_data)
    print len(label_data)
