!pip install -q nltk gensim scikit-learn

import nltk
from nltk.tokenize import TreebankWordTokenizer
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# === Set up tokenizer (avoids punkt_tab issue) ===
tokenizer = TreebankWordTokenizer()

def preprocess(text):
    return tokenizer.tokenize(text.lower())

# === Train Word2Vec Model ===
def train_model(sentences):
    tokenized = [preprocess(sent) for sent in sentences]
    model = Word2Vec(sentences=tokenized, vector_size=100, window=5, min_count=1)
    return model

def sentence_vector(sentence, model):
    words = preprocess(sentence)
    vecs = [model.wv[w] for w in words if w in model.wv]
    return np.mean(vecs, axis=0).reshape(1, -1) if vecs else np.zeros((1, 100))

def similarity_score(sent1, sent2, model):
    vec1 = sentence_vector(sent1, model)
    vec2 = sentence_vector(sent2, model)
    sim = cosine_similarity(vec1, vec2)[0][0]
    verdict = "Plagiarized" if sim > 0.7 else "Not Plagiarized"
    return sim, verdict

# === Sample Inputs ===
sent1 = "The quick brown fox jumps over the lazy dog."
sent2 = "A fast brown fox leaps over a sleepy dog."

model = train_model([sent1, sent2])
score, verdict = similarity_score(sent1, sent2, model)

print(f"Similarity Score: {score:.2f}")
print(f"Verdict: {verdict}")
