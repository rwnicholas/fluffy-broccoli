#!/usr/bin/python3

from pathlib import Path
from typing_extensions import final
from tokenizers import ByteLevelBPETokenizer
import cluster
import pandas as pd
import gtinFixer as gtf

def preprocessing(text):
    finalText = text.lower()
    finalText = finalText.replace(".", " ")
    return finalText

data = pd.read_csv("data/produtos.csv", dtype={'descp': str})
data['gtin'] = data['gtin'].apply(lambda x: gtf.valida_gtin(str(x)))
data['descp'] = data['descp'].apply(lambda x: preprocessing(x))
data = data.dropna()
data = data.reset_index(drop=True)

sentences = data['descp']
sentences.to_csv("data/text_data.txt", index=False)

