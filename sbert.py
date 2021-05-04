#!/usr/bin/python3

from sentence_transformers import SentenceTransformer
import pandas as pd
data = pd.read_csv("data/produtos.csv")
print(data.head())


sentences = data['descp']

# model = SentenceTransformer('neuralmind/bert-large-portuguese-cased')
model = SentenceTransformer('stsb-roberta-base')
embeddings = model.encode(sentences, show_progress_bar=True)

for sentence, embedding in zip(sentences, embeddings):
    print("Sentence:", sentence)
    print("Embedding:", embedding)
    print(len(embedding))
    print("")