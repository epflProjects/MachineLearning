#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from collections import defaultdict
import re
import csv
from helpers import load_data_and_labels, get_embeddings, clean_str
from models import get_model_simple_convolution, get_model_paper_convolution

import sys
import os

os.environ['KERAS_BACKEND']= "tensorflow"
# theano'

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout
from keras.models import Model
from keras.models import model_from_json

### PARAMETERS
# TODO better fix of these numbers
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 200
VALIDATION_SPLIT = 0.2 # TODO try 0.5?

PREDICT = True

positive_data = "./data/train_pos.txt"
negative_data = "./data/train_neg.txt"

print("Loading data...")
x_train, labels = load_data_and_labels(positive_data, negative_data)

source_test = open("./data/test_data.txt", "r")

tests = list()
for r in source_test:
    r = r.partition(",")[2]
    tests.append(r)

tests = [s.strip() for s in tests]
x_test = [clean_str(sent) for sent in tests]

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(x_train)
sequences = tokenizer.texts_to_sequences(x_test)


word_index = tokenizer.word_index
embeddings_index = get_embeddings("glove.twitter.27B.200d.txt")
embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(len(word_index) + 1, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=True)

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
l_cov1 = Conv1D(128, 5, activation='relu')(embedded_sequences)
l_pool1 = MaxPooling1D(5)(l_cov1)
l_cov2 = Conv1D(128, 5, activation='relu')(l_pool1)
l_pool2 = MaxPooling1D(5)(l_cov2)
l_cov3 = Conv1D(128, 5, activation='relu')(l_pool2)
l_pool3 = MaxPooling1D(35)(l_cov3)  # global max pooling
l_flat = Flatten()(l_pool3)
l_dense = Dense(128, activation='relu')(l_flat)
preds = Dense(2, activation='softmax')(l_dense)

loaded_model = get_model_simple_convolution(embeddings_index, word_index, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM)
loaded_model.load_weights("./runs/simpleModel.h5")
print("Calculation of the predictions")
print('Found %s unique tokens.' % len(word_index))

x_test = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

z = loaded_model.predict(x_test, verbose=1)
# print(z)
prediction = np.argmax(z, axis=-1)
with open("data/predictions.csv", "w") as prediction_file:
    wtr = csv.writer(prediction_file)
    wtr.writerow(("Id", "Prediction"))
    line = 1
    for pred in prediction:
        if pred == 0:
            pred = str(-1)
        else:
            pred = str(1)
        wtr.writerow((line, pred))
        line += 1
