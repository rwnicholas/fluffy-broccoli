#!/usr/bin/python3

import pandas as pd
import gtinFixer as gtf
import fasttext
import cluster
import numpy as np

data = pd.read_csv("data/produtos.csv")
data['gtin'] = data['gtin'].apply(lambda x: gtf.valida_gtin(str(x)))
data = data.dropna()
data = data.reset_index(drop=True)

print("População:", data['gtin'].count())

k = data['gtin'].nunique(dropna=True)

print("Número de Clusters:", k)

model = fasttext.load_model('data/crawl-300d-2M-subword.bin')

data['vec'] = data['descp'].apply(lambda x: model.get_sentence_vector(x))
data['vec'] = data['vec'].apply(lambda x: x.reshape([-1]))
tmpArray = data['vec'].to_numpy()

embeddings = np.empty((data['gtin'].count(),300), dtype=object)
for i in range(len(embeddings)):
    embeddings[i] = tmpArray[i]

data = data.drop(columns='vec')
cluster.clustering(k, embeddings, data)