import faiss
import numpy as np

class FaissKMeans:
    def __init__(self, n_clusters=8, n_init=1, max_iter=50):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.kmeans = None
        self.cluster_centers_ = None
        self.inertia_ = None
        self.labels_ = None
        self.obj = None

    def predict(self, X):
        return self.kmeans.index.search(X.astype(np.float32), 1)[1]

    def fit(self, X, ngpu=2):
        # This code is based on https://github.com/facebookresearch/faiss/blob/master/benchs/kmeans_mnist.py
        D = X.shape[1]
        clus = faiss.Clustering(D, self.n_clusters)
        
        # otherwise the kmeans implementation sub-samples the training set
        clus.max_points_per_centroid = 10000000
        
        clus.niter = self.max_iter
        clus.nredo = self.n_init
        
        res = [faiss.StandardGpuResources() for i in range(ngpu)]

        flat_config = []
        for i in range(ngpu):
            cfg = faiss.GpuIndexFlatConfig()
            cfg.useFloat16 = False
            cfg.device = i
            flat_config.append(cfg)

        if ngpu == 1:
            index = faiss.GpuIndexFlatL2(res[0], D, flat_config[0])
        else:
            indexes = [faiss.GpuIndexFlatL2(res[i], D, flat_config[i])
                    for i in range(ngpu)]
            index = faiss.IndexProxy()
            for sub_index in indexes:
                index.addIndex(sub_index)

        clus.train(X.astype(np.float32), index)

        self.labels_ = index.search(X.astype(np.float32), 1)[1].reshape(-1)

        stats = clus.iteration_stats
        stats = [stats.at(i) for i in range(stats.size())]
        self.obj = np.array([st.obj for st in stats])

        self.kmeans = clus
        self.inertia_ = self.obj[-1]
        self.cluster_centers_ = self.kmeans.centroids