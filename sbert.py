#!/usr/bin/python3

from pandas.core.frame import DataFrame
from sentence_transformers import SentenceTransformer
# from sklearn.cluster import KMeans
from FKMeans import FaissKMeans
import pandas as pd
import gtinFixer as gtf
import validateCluster as valCluster

data = pd.read_csv("data/produtos.csv")
data['gtin'] = data['gtin'].apply(lambda x: gtf.valida_gtin(str(x)))
data = data.dropna()
data = data.reset_index(drop=True)

print("População:", data['gtin'].count())

k = data['gtin'].nunique(dropna=True)

print("Número de Clusters:", k)

sentences = data['descp']

model = SentenceTransformer('neuralmind/bert-large-portuguese-cased', device='cuda')
# model = SentenceTransformer('paraphrase-xlm-r-multilingual-v1', device='cuda')
# model = SentenceTransformer('paraphrase-distilroberta-base-v1', device='cuda')

embeddings = model.encode(sentences, show_progress_bar=True)

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