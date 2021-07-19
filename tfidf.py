#!/usr/bin/python3

from FKMeans import FaissKMeans
from sklearn.cluster import KMeans
import cluster
import numpy as np
import pandas as pd
import gtinFixer as gtf
import validateCluster as valCluster
from sklearn.feature_extraction.text import HashingVectorizer

data = pd.read_csv("data/produtos.csv")
data['gtin'] = data['gtin'].apply(lambda x: gtf.valida_gtin(str(x)))
data = data.dropna()
data = data.reset_index(drop=True)

print("População:", data['gtin'].count())

k = data['gtin'].nunique(dropna=True)

print("Número de Clusters:", k)

sentences = data['descp']

vectorizer = HashingVectorizer(n_features=2**12)
response = vectorizer.transform(sentences)
cluster.gpu_clustering(k, response.todense(), data)