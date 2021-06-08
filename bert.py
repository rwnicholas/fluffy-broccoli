#!/usr/bin/python3

from transformers import AutoTokenizer, AutoModel
import torch
import cluster
import pandas as pd
import gtinFixer as gtf

data = pd.read_csv("data/produtos.csv", dtype={'descp': str})
data['gtin'] = data['gtin'].apply(lambda x: gtf.valida_gtin(str(x)))
data['descp'] = data['descp'].apply(lambda x: cluster.preprocessing(x))
data = data.dropna()
data = data.reset_index(drop=True)

print("População:", data['gtin'].count())

k = data['gtin'].nunique(dropna=True)

print("Número de Clusters:", k)
#.to_numpy().reshape((-1,1))
sentences = data['descp']
print(sentences.to_numpy())

# tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-distilroberta-base-v2', do_lower_case=True)
# model = AutoModel.from_pretrained('sentence-transformers/paraphrase-distilroberta-base-v2')

# encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# #Compute token embeddings
# with torch.no_grad():
#     model_output = model(**encoded_input)
#     embeddings = model_output[0][:,0] #Take the first token ([CLS]) from each sentence 

# # cluster.clustering(k, embeddings, data)