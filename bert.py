#!/usr/bin/python3

from transformers import AutoTokenizer, AutoModel

sentences = ['Eu não amo o brasil, caralho, mas vivo aqui.']

tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')
model = AutoModel.from_pretrained('neuralmind/bert-base-portuguese-cased')

inputs = tokenizer(sentences, return_tensors="pt")
outputs = model(**inputs)

for sentence, embedding in zip(sentences, outputs):
    print("Sentence:", sentence)
    print("Embedding:", embedding[0][0])
    print(len(embedding))