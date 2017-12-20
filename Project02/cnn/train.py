#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from helpers import load_data_and_labels, get_embeddings
from models import get_model_simple_convolution, get_model_paper_convolution, get_model_paper_2_convolution

import os
import argparse

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

os.environ['KERAS_BACKEND'] = "tensorflow"
# theano'

### PARAMETERS
# TODO better fix of these numbers
MAX_SEQUENCE_LENGTH = 1000 # TODO how to fix?
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 200
VALIDATION_SPLIT = 0.04

PREDICT = False

positive_data = "./data/train_pos.txt"
negative_data = "./data/train_neg.txt"

print("Loading data...")
x_text, labels = load_data_and_labels(positive_data, negative_data)

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(x_text)
sequences = tokenizer.texts_to_sequences(x_text)

word_index = tokenizer.word_index
print("Found", len(word_index), "unique tokens.")

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

print('Number of positive and negative reviews in training and validation set')
print(y_train.sum(axis=0))
print(y_val.sum(axis=0))

print("Importing embeddings...")
embeddings_index = get_embeddings("glove.twitter.27B.200d.txt")

#model = get_model_simple_convolution(embeddings_index, word_index, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM)
# model = get_model_paper_convolution(embeddings_index, word_index, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM)
model = get_model_paper_2_convolution(embeddings_index, word_index, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM)

print("model fitting - simplified convolutional neural network")
model.summary()
# model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=128)
# model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=50)
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=64)

print("saving model on disk")
model.save("./runs/paper2Model.h5")
model.save_weights("./runs/weights_paper2Model.h5")
