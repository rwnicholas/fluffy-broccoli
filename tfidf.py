#!/usr/bin/python3

from FKMeans import FaissKMeans
from sklearn.cluster import KMeans
import cluster
import numpy as np
import pandas as pd
import gtinFixer as gtf
import validateCluster as valCluster
from sklearn.feature_extraction.text import TfidfVectorizer

data = pd.read_csv("data/produtos.csv")
data['gtin'] = data['gtin'].apply(lambda x: gtf.valida_gtin(str(x)))
data = data.dropna()
data = data.reset_index(drop=True)

print("População:", data['gtin'].count())

k = data['gtin'].nunique(dropna=True)

print("Número de Clusters:", k)

sentences = data['descp']

vectorizer = TfidfVectorizer()
response = vectorizer.fit_transform(sentences)

clustering_model = KMeans(n_clusters=k, n_init=1, max_iter=50)
clustering_model.fit(response)
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