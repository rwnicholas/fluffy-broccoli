#!/usr/bin/python3

from sentence_transformers import SentenceTransformer
from FKMeans import FaissKMeans
import cluster
import pandas as pd
import gtinFixer as gtf
import validateCluster as valCluster

data = pd.read_csv("data/produtos.csv")
data['gtin'] = data['gtin'].apply(lambda x: gtf.valida_gtin(str(x)))
data = data.dropna()
data = data.reset_index(drop=True)
data = data.sample(2)
data = data.reset_index(drop=True)

print("População:", data['gtin'].count())

k = data['gtin'].nunique(dropna=True)

print("Número de Clusters:", k)

sentences = data['descp']

model = SentenceTransformer('paraphrase-distilroberta-base-v2', device='cuda')

embeddings = model.encode(sentences, show_progress_bar=True)
print(len(embeddings[0]))
# cluster.clustering(k, embeddings, data)