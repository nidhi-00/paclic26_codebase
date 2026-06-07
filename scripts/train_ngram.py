# scripts/train_ngram.py
from nltk.lm import KneserNeyInterpolated
from nltk.lm.preprocessing import padded_everygram_pipeline

order = 5
sentences = [line.strip().split() for line in open("data/raw/lm_train.txt") if line.strip()]
ngrams, vocab = padded_everygram_pipeline(order, sentences)
model = KneserNeyInterpolated(order=order)
model.fit(ngrams, vocab)
