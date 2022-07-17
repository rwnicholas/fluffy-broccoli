from time import time
import pandas as pd
from FKMeans import FaissKMeans
from sklearn.cluster import KMeans as skKMeans
import validateCluster as valCluster
import os, glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from sklearn.metrics import adjusted_rand_score, homogeneity_score, silhouette_score, completeness_score, v_measure_score, davies_bouldin_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.preprocessing import LabelEncoder
from validclust import dunn

def preprocessing(text):
    # unitData = ['gr', 'g', 'grama', 'gramas', 'kg', 'kgs', 'quilo', 'quilos', 'quilograma', 'quilogramas', 'kilo', 'kilos', 'kilograma', 'kilogramas', 'm', 'metro', 'metros', 'cm', 'centímetro', 'centimetro', 'l', 'litro', 'litros', 'ml', 'mililitros', 'caixa', 'caixas', 'cx', 'cxs', 'balde', 'baldes', 'sache', 'pacote', 'pacotes', 'pc', 'pct', 'pcs', 'pcts', 'c/', 'com', 'folha', 'folhas', 'fl', 'fls', 'sachê', 'saches', 'sachês', 'envelope', 'envelopes', 'env', 'envs', 'saco', 'sacos', 'un', 'qtd', 'und', 'cxt']
    # text = text.lower()
    # text = list([val for val in text
    #                 if val.isalpha() or val == " "])
    # text = "".join(text)
    # # Convert all numbers in the article to the word 'num' using regular expressions
    # text = text.rstrip()
    # text = text.lstrip()
    # text = re.sub('\s{2,}', ' ', text)
    # for u in unitData:
    #     text = text.replace((' '+u+' '), " ")
    finalText = text.lower()
    finalText = finalText.replace(".", " ")
    return text

def find_k(k_, embeddings, _kmeans = FaissKMeans):
    K = [*range(500, (2*10**4), 500)]
    K.append(k_)
    K.sort()

    inertias = []

    for k in K:
        clustering_model = _kmeans(n_clusters=k, n_init=1, max_iter=50)
        clustering_model.fit(embeddings)
        labels = clustering_model.labels_
        inertias.append(silhouette_score(embeddings, labels, metric = 'euclidean'))
        # print(k)

    plt.plot(K, inertias, 'bx-')
    plt.xlabel('Values of K')
    plt.ylabel('Silhouette Score')
    plt.title('The Silhouette Score')
    plt.xticks(K, rotation=75, fontsize=10)
    plt.savefig("silhouette.png")
    plt.close()

def cpu_clustering(k_, embeddings_, data_, keep_clusters = False):
    return __clustering(k_, embeddings_, data_, skKMeans, keep_clusters)

def gpu_clustering(k_, embeddings_, data_, keep_clusters = False):
    return __clustering(k_, embeddings_, data_, FaissKMeans, keep_clusters)

def __clustering(k, embeddings, data, kmeans_, keep_clusters):
    clustering_model = kmeans_(n_clusters=k, n_init=1, max_iter=50)
    clustering_model.fit(embeddings)
    cluster_assignment = clustering_model.labels_

    clustered_sentences = [[] for i in range(k)]
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        clustered_sentences[cluster_id].append(data.loc[[sentence_id]])

    if not os.path.exists("clusters/"):
        os.mkdir("clusters/")

    countK = 0
    for cluster in clustered_sentences:
        if len(cluster) == 0:
            continue
        result_conc = pd.concat(cluster)
        result_conc.to_csv("clusters/cluster" + str(countK) + ".csv", index=False)
        countK+=1

    print("Acurácia média:", valCluster.accClusters(countK))

    if keep_clusters == False:
        files = glob.glob('clusters/*')
        for f in files:
            os.remove(f)

    # return homogeneity_score(LabelEncoder().fit_transform(data['gtin']), clustering_model.labels_)
    # return f1_score(LabelEncoder().fit_transform(data['gtin']), clustering_model.labels_, average='weighted')
    # return accuracy_score(LabelEncoder().fit_transform(data['gtin']), clustering_model.labels_)
    # return precision_recall_curve(LabelEncoder().fit_transform(data['gtin']), clustering_model.labels_)
    dist = pairwise_distances(embeddings)

    return {
        'finalstamp': time(),
        'davies_bouldin': davies_bouldin_score(embeddings, clustering_model.labels_),
        'dunn_index': dunn(dist, clustering_model.labels_),
        'contingency_matrix': contingency_matrix(LabelEncoder().fit_transform(data['gtin']), clustering_model.labels_),
        'homogeneity': homogeneity_score(LabelEncoder().fit_transform(data['gtin']), clustering_model.labels_),
        'adjusted_rand': adjusted_rand_score(LabelEncoder().fit_transform(data['gtin']), clustering_model.labels_),
        'completeness': completeness_score(LabelEncoder().fit_transform(data['gtin']), clustering_model.labels_),
        'v_measure': v_measure_score(LabelEncoder().fit_transform(data['gtin']), clustering_model.labels_)
    }