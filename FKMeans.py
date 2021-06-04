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

    # def fit(self, X):
    #     self.kmeans = faiss.Kmeans(d=X.shape[1],
    #                                k=self.n_clusters,
    #                                niter=self.max_iter,
    #                                nredo=self.n_init)
    #     self.kmeans.train(X.astype(np.float32))
    #     self.cluster_centers_ = self.kmeans.centroids
    #     self.inertia_ = self.kmeans.obj[-1]

    def predict(self, X):
        return self.kmeans.index.search(X.astype(np.float32), 1)[1]

    def run_faiss_gpu(self, X, ngpu):
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
                
        
        # Run clustering
        clus.train(X, index)

        self.labels_ = index.search(X, 1)[1].reshape(-1)

        self.kmeans = clus