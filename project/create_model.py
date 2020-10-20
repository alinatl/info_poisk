import gensim
from gensim.models import Word2Vec, KeyedVectors

model_file = '/content/araneum_none_fasttextcbow_300_5_2018.model'
model = KeyedVectors.load(model_file)

import nltk
nltk.download('stopwords')
import re
import json
import os
import numpy as np
import pymorphy2
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from math import log
import collections
import sklearn
from collections import Counter

from nltk.corpus import stopwords
stop_words = stopwords.words('russian')
# stop_words.extend()

morph = pymorphy2.MorphAnalyzer()
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()



answers = pd.read_csv('answers_base.csv', encoding = 'windows-1251', sep = ';')
queries = pd.read_csv('queries_base.csv', encoding = 'windows-1251', sep = ';')

ans = answers[['Номер связки','Текст вопросов']].dropna(axis = 0, how ='any')
qw = queries[['Текст вопроса', 'Номер связки\n']].dropna(axis = 0, how ='any')
qw.rename(columns={'Текст вопроса': 'Текст вопросов', 'Номер связки\n': 'Номер связки'}, inplace=True)
train = pd.concat([ans, qw.iloc[0:int(qw.shape[0]*0.7), :]]) #train
train['idx'] = train.reset_index().index

queries2 = qw.iloc[int(qw.shape[0]*0.7):, 0].tolist()
test = dict(zip(qw.iloc[int(qw.shape[0]*0.7):, 0], qw.iloc[int(qw.shape[0]*0.7):, 1])) # test

def create_corpus(train_text, NER = False, fun = False):
  corpus = []
  for question in train_text['Текст вопросов']:
      question = question.replace('\n', ' ').replace('/', ' ')
      if NER == False:
        pass
      else:
        question = fun(question)
      words_doc = tokenize_ru(question)
      corpus.append(words_doc)
  return corpus

def tokenize_ru(sentence):
    sentence = sentence.replace('\n', ' ').replace('/', ' ')
    sentence = re.sub(r'[\'"”\,\!\?\.\-\(\)\[\]\:\;\»\«\>\—]', ' ', str(sentence).rstrip("']"))
    sentence = re.sub(r'[0-9]', ' ', str(sentence))
    sentence = sentence.lower()
    tokens = sentence.split()
    tokens = [i for i in tokens if (i not in stop_words)]
    tokens = [morph.parse(i)[0].normal_form for i in tokens]
    tokens = ' '.join(tokens)
    return tokens

# создает корпус w2v

def make_w2v_doc_matr(corpus):
    all_vectors = []
    for doc in corpus:
        lemmas = doc.split(' ')

        lemmas_vectors = np.zeros((len(lemmas), model.vector_size))
        vec = np.zeros((model.vector_size,))

        for idx, lemma in enumerate(lemmas):
            lemma = str(lemma)
            if lemma in model:
                lemmas_vectors[idx] = model[lemma]

        if lemmas_vectors.shape[0] is not 0:
            vec = np.mean(lemmas_vectors, axis=0)
            vec = np.reshape(vec, (1, 300))
        all_vectors.append(vec / np.sqrt(np.sum (vec ** 2)))
        
    matr = np.concatenate(all_vectors)
    return matr

# создает матрицу для каждого текста в отдельности 
def create_doc_matrix(text):
    text = text.replace('\n', ' ').replace('/', ' ')
    text = tokenize_ru(text)
    lemmas = text.split(' ')

    lemmas_vectors = np.zeros((len(lemmas), model.vector_size))
    vec = np.zeros((model.vector_size,))

    for idx, lemma in enumerate(lemmas):
        if lemma in model:
            vec = model[lemma]
            lemmas_vectors[idx] = vec / np.sqrt(np.sum (vec ** 2))
            
    return lemmas_vectors

def create_mat_tfidf(corpus):
  X = vectorizer.fit_transform(corpus)
  X.toarray()
  return X

def bm25(tf_q_d, l, N, nq, corpus) -> float:
    k = 2.0
    b = 0.75
    aver = Average(corpus)
    TF = (tf_q_d * (k+1))/(tf_q_d + k*(1 - b + b*(l/aver)))
    IDF = log((N-nq+0.5)/(nq + 0.5))
    result = IDF*TF
    return result

def Average(lst):
    lst2 = []
    d = {}
    for doc in lst:
        lst2.append(len(doc.split(' ')))
    return sum(lst2) / len(lst)

def create_mat_bm25(corpus):
  N = len(corpus)
  nq = Counter(full_data)
  matr = np.zeros((N, len(nq)))

  for i, doc in enumerate(corpus):
      doc = doc.split(' ')
      tf_q_d = Counter(doc)
      l = len(doc)
      for j, word in enumerate(set(full_data)):
          if tf_q_d[word] == 0:
              matr[i, j] = 0
          else:
              matr[i, j] = bm25(tf_q_d[word], l, N, nq[word], corpus)
  return matr

corpus = create_corpus(train)
full_data = ' '.join(corpus).split(' ')

X_tfidf = create_mat_tfidf(corpus)
with open('X_tfidf.pickle', 'wb') as f:
  pickle.dump(X_tfidf, f)

X_bm25 = create_mat_bm25(corpus)
with open('X_bm25.pickle', 'wb') as f:
  pickle.dump(X_bm25, f)

X_w2v = make_w2v_doc_matr(corpus)
with open('X_w2v.pickle', 'wb') as f:
  pickle.dump(X_w2v, f)

corpus_matr = []
for doc in corpus:
  m = create_doc_matrix(doc)
  corpus_matr.append(m)
with open('corpus_matr.pickle', 'wb') as f:
  pickle.dump(corpus_matr, f)

X = vectorizer.fit(corpus)
with open('vectorizer.pickle', 'wb') as f:
  pickle.dump(X, f)

corpus = create_corpus(train)
full_data = ' '.join(corpus).split(' ')
with open('full_data.pickle', 'wb') as f:
  pickle.dump(full_data, f)
