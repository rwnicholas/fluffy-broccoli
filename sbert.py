#!/usr/bin/python3

from time import time
from pandas.core.frame import DataFrame
from sentence_transformers import SentenceTransformer
import cluster
import pandas as pd
import gtinFixer as gtf
import matplotlib.pyplot as plt

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
print(cluster.gpu_clustering(k, embeddings, data, keep_clusters=False))

# cluster.find_k(k, embeddings)

# # Homogeneity Test
# K = [*range(100, (2*10**4), 500)]
# K.append(k)
# # K.remove(80100)
# K.sort()

# homogeneity_all = []
# time_all = []
# savingDataList = []


# for k_ in K:
#     start = time()

#     homogeneity = cluster.gpu_clustering(k_, embeddings, data, keep_clusters=False)
#     homogeneity_all.append(homogeneity)

#     end = time()

#     run_time = end - start
#     time_all.append(run_time)

#     tmpDict = {
#         'K Value': k_,
#         'Homogeneity': homogeneity,
#         'Runtime': run_time
#     }
#     savingDataList.append(tmpDict)

# # Saving results
# savingDataframe = DataFrame(savingDataList)
# savingDataframe.to_csv("data/homogeneity_runtime_results_al.csv", index=False)