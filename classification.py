from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import numpy as np

#load dataset
news_20 = fetch_20newsgroups(subset='train', shuffle=True, random_state=42)

#cal_document_vector
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(news_20.data)
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)

#train classifier
clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)


news_20_test = fetch_20newsgroups(subset='test', shuffle=True, random_state=42)
docs_test = news_20_test.data
predicted = count_vect.fit_transform(docs_test)

