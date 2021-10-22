from scipy.spatial import KDTree


class NearestNeighbors(object):
    # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html
    # TODO: approximate KDTrees
    # https://github.com/lmcinnes/pynndescent
    # https://github.com/spotify/annoy
    # https://github.com/flann-lib/flann
    def __init__(self, data=[], embed_fn=lambda x: x, **kwargs):
        # TODO: maintain tree and brute-force list
        self.data = [] # TODO: self.kd_tree.data
        self.kd_tree = None
        self.embed_fn = embed_fn
        self.kwargs = kwargs
        self.add_data(data)
    def add_data(self, data):
        indices = []
        for x in data:
            indices.append(len(self.data))
            self.data.append(x)
        if not self.data:
            return zip(indices, data)
        embedded_data = list(map(self.embed_fn, self.data))
        self.kd_tree = KDTree(embedded_data,
                              #leafsize=10, compact_nodes=True, copy_data=False, balanced_tree=True, boxsize=None
                              **self.kwargs,
                              )
        return zip(indices, data)
    def query_neighbors(self, x, **kwargs):
        # TODO: class **kwargs
        embedded = self.embed_fn(x)
        # k=1, eps=0, p=2, distance_upper_bound=inf, workers=1
        distances, indices = self.kd_tree.query(embedded, **kwargs)
        return [(d, i, self.data[i]) for d, i in zip(distances, indices)]