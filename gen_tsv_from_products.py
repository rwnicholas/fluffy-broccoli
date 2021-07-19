#!/usr/bin/python3

from sentence_transformers import SentenceTransformer, util
import cluster
import pandas as pd
import gtinFixer as gtf
import csv

data = pd.read_csv("data/produtos.csv", dtype={'gtin': str})
data['gtin'] = data['gtin'].apply(lambda x: gtf.valida_gtin(str(x)))
data = data.dropna()
data = data.reset_index(drop=True)
data['descp'] = data['descp'].str.lower()
data['descp'] = data['descp'].apply(lambda x: " ".join(x.split()))
data = data[['gtin', 'descp']]

groups = data.groupby('gtin')
# print(groups.groups)
with open('./data/products_similar.tsv', 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t', quoting=csv.QUOTE_ALL)
    tsv_writer.writerow(['sentence1', 'sentence2'])
    for group in groups:
        if len(group[1]) > 1:
            for x in group[1]['descp'].iloc[1:]:
                tsv_writer.writerow([group[1]['descp'].iloc[0], x])
