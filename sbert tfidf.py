#!/usr/bin/python3

from sentence_transformers import SentenceTransformer, util
import cluster
import pandas as pd
import gtinFixer as gtf
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

tfidf_model = TfidfVectorizer()

def get_most_important(sentence):
    response = tfidf_model.transform([sentence])
    feature_array = np.array(tfidf_model.get_feature_names())
    tfidf_sorting = np.argsort(response.toarray()).flatten()[::-1]

    n = 3
    top_n = feature_array[tfidf_sorting][:n]
    return top_n


data = pd.read_csv("data/produtos.csv")
data['gtin'] = data['gtin'].apply(lambda x: gtf.valida_gtin(str(x)))
data = data.dropna()
data = data.reset_index(drop=True)

print("População:", data['gtin'].count())

k = data['gtin'].nunique(dropna=True)

print("Número de Clusters:", k)


sentences = data['descp']
tfidf = tfidf_model.fit_transform(sentences)

print("comecei")
sentences = sentences.apply(lambda x: get_most_important(x))
# print("terminei")

# model = SentenceTransformer('paraphrase-distilroberta-base-v2', device='cuda')

# embeddings = model.encode(sentences, show_progress_bar=True)

# cluster.gpu_clustering(k, embeddings, data)