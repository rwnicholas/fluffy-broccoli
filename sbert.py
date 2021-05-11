#!/usr/bin/python3

from pandas.core.frame import DataFrame
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import pandas as pd

data = pd.read_csv("data/produtos14.csv")

sentences = data['descp']

model = SentenceTransformer('neuralmind/bert-large-portuguese-cased')
# model = SentenceTransformer('stsb-distilbert-base')

embeddings = model.encode(sentences, show_progress_bar=True)

k = 8273
# k = 5

clustering_model = KMeans(n_clusters=k)
clustering_model.fit(embeddings)
cluster_assignment = clustering_model.labels_

clustered_sentences = [[] for i in range(k)]
for sentence_id, cluster_id in enumerate(cluster_assignment):
    clustered_sentences[cluster_id].append(data.loc[[sentence_id]])

for i, cluster in enumerate(clustered_sentences):
    print("Cluster ", i+1)
    result_conc = pd.concat(cluster)
    result_conc.to_csv("clusters/cluster" + str(i) + ".csv", index=False)
