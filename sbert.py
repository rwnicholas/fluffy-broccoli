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
# print(cluster.gpu_clustering(k, embeddings, data, keep_clusters=False))



# Adjusted Rand Index
K = [*range(100, k+(3*10**4), 5000)]
K.append(k)
K.remove(80100)
K.sort()

adjusted_rand_index_all = []
time_all = []
savingDataList = []


for k_ in K:
    start = time()

    adjusted_rand_index = cluster.gpu_clustering(k_, embeddings, data, keep_clusters=False)
    adjusted_rand_index_all.append(adjusted_rand_index)

    end = time()

    run_time = end - start
    time_all.append(run_time)

    tmpDict = {
        'K Value': k_,
        'Adjusted Rand Index': adjusted_rand_index,
        'Runtime': run_time
    }
    savingDataList.append(tmpDict)

# Saving results
savingDataframe = DataFrame(savingDataList)
savingDataframe.to_csv("data/adjusted_rand_index_runtime_results.csv", index=False)

# Ploting results
plt.plot(K, adjusted_rand_index_all, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Adjusted Rand Index')
plt.title('Adjusted Rand Index for numbers of clusters')
plt.xticks(K, rotation=75, fontsize=10)
plt.margins(x=0)
plt.tight_layout()
plt.savefig("adjusted_rand_index_graph.png")
plt.close()

plt.plot(K, time_all, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Time (seconds)')
plt.title('Runtime for numbers of clusters')
plt.xticks(K, rotation=75, fontsize=10)
plt.margins(x=0)
plt.tight_layout()
plt.savefig("runtime_graph.png")
plt.close()