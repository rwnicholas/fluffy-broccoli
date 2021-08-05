import pandas as pd
from FKMeans import FaissKMeans
from sklearn.cluster import KMeans as skKMeans
import validateCluster as valCluster
import os, glob

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

def cpu_clustering(k_, embeddings_, data_):
    __clustering(k_, embeddings_, data_, skKMeans)

def gpu_clustering(k_, embeddings_, data_):
    __clustering(k_, embeddings_, data_, FaissKMeans)

def __clustering(k, embeddings, data, kmeans_, keep_clusters = False):
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