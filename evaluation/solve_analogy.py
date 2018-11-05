# -*- coding: utf-8 -*-

"""
 Simple example showing answering analogy questions
"""
import logging
from datasets.analogy import fetch_google_analogy
from embeddings import fetch_GloVe

# Configure logging
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')

# Fetch skip-gram trained on GoogleNews corpus and clean it slightly
w = fetch_GloVe(corpus="wiki-6B", dim=50)

# Fetch analogy dataset
data = fetch_google_analogy()

for cat in (set(data.category)):
    print(cat)

# Pick a sample of data and calculate answers
subset = [50, 1000, 4000, 10000, 14000]
for id in subset:
    w1, w2, w3 = data.X[id][0], data.X[id][1], data.X[id][2]
    print("Question: {} is to {} as {} is to ?".format(w1, w2, w3))
    print("Answer: " + data.y[id])
    print("Predicted: " + " ".join(w.nearest_neighbors(w[w2] - w[w1] + w[w3], exclude=[w1, w2, w3])))

