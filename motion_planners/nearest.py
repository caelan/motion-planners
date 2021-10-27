from scipy.spatial import KDTree
from itertools import product

import numpy as np
import math

NORMAL = [0]
UNIT = [-1, 0, +1]
CIRCULAR = [-2*math.pi, 0, +2*math.pi]


class NearestNeighbors(object):
    # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html
    # TODO: approximate KDTrees
    # https://github.com/lmcinnes/pynndescent
    # https://github.com/spotify/annoy
    # https://github.com/flann-lib/flann
    def __init__(self, data=[], circular=[], embed_fn=lambda x: x, **kwargs): # [0, 1]
        # TODO: maintain tree and brute-force list
        self.data = [] # TODO: self.kd_tree.data
        self.equivalent = []
        self.embedded = []
        self.kd_tree = None
        self.circular = circular
        self.embed_fn = embed_fn
        self.kwargs = kwargs
        self.add_data(data)
    def add_data(self, new_data):
        indices = []
        for x in new_data:
            index = len(self.data)
            indices.append(index)
            self.data.append(x)
            domains = [UNIT if (k in self.circular) else NORMAL for k in range(len(x))]
            for dx in product(*domains):
                wx = x + np.array(dx)
                self.equivalent.append(index)
                self.embedded.append(self.embed_fn(wx))

        if self.embedded:
            self.kd_tree = KDTree(self.embedded,
                                  #leafsize=10, compact_nodes=True, copy_data=False, balanced_tree=True, boxsize=None
                                  **self.kwargs)
        return zip(indices, new_data)
    def query_neighbors(self, x, **kwargs):
        # TODO: class **kwargs
        embedded = self.embed_fn(x)
        #print(x, embedded)
        # k=1, eps=0, p=2, distance_upper_bound=inf, workers=1
        distances, indices = self.kd_tree.query(embedded, **kwargs)
        return [(d, self.equivalent[i], self.data[self.equivalent[i]])
                for d, i in zip(distances, indices)] # TODO: filter
