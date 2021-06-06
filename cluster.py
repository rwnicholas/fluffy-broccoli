import pandas as pd
from FKMeans import FaissKMeans
import validateCluster as valCluster

def clustering(k, embeddings, data):
    clustering_model = FaissKMeans(n_clusters=k)
    clustering_model.run_faiss_gpu(embeddings, ngpu=2)
    cluster_assignment = clustering_model.labels_

    clustered_sentences = [[] for i in range(k)]
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        clustered_sentences[cluster_id].append(data.loc[[sentence_id]])

    countK = 0
    for cluster in clustered_sentences:
        if len(cluster) == 0:
            continue
        result_conc = pd.concat(cluster)
        result_conc.to_csv("clusters/cluster" + str(countK) + ".csv", index=False)
        countK+=1


    print("Acurácia média:", valCluster.accClusters(countK))