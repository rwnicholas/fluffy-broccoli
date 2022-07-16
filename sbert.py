#!/usr/bin/python3.7

from time import time
from pandas.core.frame import DataFrame
from sentence_transformers import SentenceTransformer
import cluster
import pandas as pd
import gtinFixer as gtf
import matplotlib.pyplot as plt
import pickle
import faiss

data = pd.read_csv("data/produtos.csv")
data['gtin'] = data['gtin'].apply(lambda x: gtf.valida_gtin(str(x)))
data = data.dropna()
data = data.reset_index(drop=True)
# data = data.sample(1000, random_state=394)
# data = data.reset_index(drop=True)

print("População:", data['gtin'].count())

k = data['gtin'].nunique(dropna=True)

print("Número de Clusters:", k)

sentences = data['descp']

model = SentenceTransformer('./paraphrase-distilroberta-base-sefaz', device='cuda')

embeddings = model.encode(sentences, show_progress_bar=True)
# clustering_model = cluster.gpu_clustering(k, embeddings, data, keep_clusters=False)

def testing(test_name):
    K = [*range(100, (1*10**5), 2500)]
    K.append(k)
    K.sort()

    test_all = []
    time_all = []
    savingDataList = []


    for k_ in K:
        start = time()

        test = cluster.gpu_clustering(k_, embeddings, data, keep_clusters=False)

        tmpDict = {
            'K Value': k_,
            'precision': test['precision'],
            'recall': test['recall'],
            'f1_score': test['f1_score'],
            'homogeneity': test['homogeneity'],
            'adjusted_rand': test['adjusted_rand'],
            'completeness': test['completeness'],
            'v_measure': test['v_measure'],
            'runtime': start - test['finalstamp']
        }
        savingDataList.append(tmpDict)

    # Saving results
    savingDataframe = DataFrame(savingDataList)
    savingDataframe.to_csv("data/"+test_name+"_results.csv", index=False)

testing('Many_tests')