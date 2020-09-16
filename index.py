import re
import json
import os
import numpy as np
import pymorphy2
import pickle

from nltk.corpus import stopwords
stop_words = stopwords.words('russian')
stop_words.extend(['что', 'это', 'так', 'вот', 'быть', 'как', 'в', '—', '–', 'к', 'на', '...', 'тот', 'то', 
                  'это', 'кто-то', 'как-то'])

morph = pymorphy2.MorphAnalyzer()
from sklearn.feature_extraction.text import CountVectorizer


def index_m(matrix, list_names_dir):
    np_matrix = matrix.toarray()
    dictionary = {}
    for i, word in enumerate(vectorizer.get_feature_names()):
            dictionary[word] = [int(sum(np_matrix[:, i]))]
            for ind, doc in enumerate(np_matrix[:, i].tolist()):
                if doc != 0:
                    dictionary[word].append(list_names_dir[ind])
    return dictionary


def tokenize_ru(sentence):
    sentence = re.sub(r'[\'"”\,\!\?\.\-\(\)\[\]\:\;\»\«\>\—]', ' ', str(sentence).rstrip("']"))
    sentence = re.sub(r'[0-9A-Za-z]', '', str(sentence))
    sentence = sentence.lower()
    sentence = sentence.replace('\ufeff', '')
    tokens = sentence.split()
    tokens = [i for i in tokens if (i not in stop_words)]
    tokens = [morph.parse(i)[0].normal_form for i in tokens]
    tokens = ' '.join(tokens)
    return tokens

### _check : в коллекции должно быть около 165 файлов
d = {}
corpus = []
list_names_dir = []

curr_dir = os.getcwd()
filepath = os.path.join(curr_dir, 'friends-data')
for root, dirs, files in os.walk(filepath):
    for name in files:
        filepath_serie = os.path.join(root, name)
        with open(filepath_serie, 'rb') as f:
            file= f.read().splitlines()
            lines = []
            for line in file:
                if line == b'':
                    pass
                else:
                    line = line.decode('utf-8')
                    line = tokenize_ru(line)
                    lines.append(line)
                    list_names_dir.append(name)
            corpus.append(' '.join(lines))

vectorizer = CountVectorizer()   
X = vectorizer.fit_transform(corpus)
di = index_m(X, list_names_dir)


def get_key(d, value):
    for k, v in d.items():
        if v[0] == value:
            return k

# выдает самое популярное и редкое слово

ma = max([i[0] for i in di.values()])
mi = min([i[0] for i in di.values()])
print(get_key(di, ma))
print(get_key(di, mi), '\n')

# выводит набор слов, который есть во всех документах коллекции
for k in di.keys():
    if len(di[k])-1 > 164:
        print(k)
print()

persons = ['моника',
'рэйчел',
'чендлер',
'фиби',
'росс',
'джоуя']

# выводит максимальное значение 
name_max_val = 0
name_pop = ''
for per in persons:
    if di[per][0] > name_max_val:
        name_max_val = di[per][0]
        name_pop = per
print(name_pop)

with open("data_file.json", "w") as write_file:
    json.dump(di, write_file)

with open('data_file_dict.pickle', 'wb') as write_file:
    pickle.dump(di, write_file)
    
with open('data_file_mtx.pickle', 'wb') as write_file:
    pickle.dump(X, write_file)   
