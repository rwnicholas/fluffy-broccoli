#!/usr/bin/python3.7

from pathlib import Path
from tokenizers import ByteLevelBPETokenizer

paths = [str(x) for x in Path("./data/").glob("**/*.txt")]

tokenizer = ByteLevelBPETokenizer()

tokenizer.train(files=paths, vocab_size=532433, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

tokenizer.save_model("./models/", "aquisefaz")