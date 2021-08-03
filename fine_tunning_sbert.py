import csv
from sentence_transformers import SentenceTransformer, SentencesDataset
from sentence_transformers import losses
from sentence_transformers.readers import LabelSentenceReader, InputExample
from torch.utils.data import DataLoader
# import pandas as pd

# teste = pd.read_csv('data/products_similar.tsv', delimiter='\t')
# for x in teste.isna().values:
#     if all(x):
#         print(x)

# Load pre-trained model - we are using the original Sentence-BERT for this example.
sbert_model = SentenceTransformer('paraphrase-distilroberta-base-v2', device='cuda')

# Set up data for fine-tuning 
train_samples = []
with open("data/products_similar.tsv", 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_ALL)
    for row in reader:
        inp_example = InputExample(texts=[row['sentence1'], row['sentence2']], label=0.6)
        train_samples.append(inp_example)

train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=64)
train_loss = losses.CosineSimilarityLoss(model=sbert_model)

# Fine-tune the model
sbert_model.fit(train_objectives=[(train_dataloader, train_loss)],
                epochs=1,
                scheduler='warmupcosine',
                warmup_steps=10**9,
                output_path='paraphrase-distilroberta-base-sefaz')