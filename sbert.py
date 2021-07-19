#!/usr/bin/python3

from sentence_transformers import SentenceTransformer
import cluster
import pandas as pd
import gtinFixer as gtf

data = pd.read_csv("data/produtos.csv")
data['gtin'] = data['gtin'].apply(lambda x: gtf.valida_gtin(str(x)))
data = data.dropna()
data = data.reset_index(drop=True)

print("População:", data['gtin'].count())

k = data['gtin'].nunique(dropna=True)

print("Número de Clusters:", k)

sentences = data['descp']

model = SentenceTransformer('./paraphrase-distilroberta-base-sefaz', device='cuda')

embeddings = model.encode(sentences, show_progress_bar=True)
cluster.gpu_clustering(k, embeddings, data)