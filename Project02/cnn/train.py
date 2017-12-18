import numpy as np
import pandas as pd
import json
import simplejson
from collections import defaultdict
import re
from helpers import load_data_and_labels, get_embeddings
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

# TODO better fix of these numbers
MAX_SEQUENCE_LENGTH = 140
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 200
VALIDATION_SPLIT = 0.2 # TODO try 0.5?

PREDICT = False

print("Loading data...")
x_text, labels = load_data_and_labels("./data/preprocess_train_pos_full.txt", "./data/preproceess_train_neg_full.txt")

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

model = get_model_simple_convolution(embeddings_index, word_index, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
# print("Saving the model to disk in a JSON format")
# model_json = model.to_json()
# with open('data/convModel.json', 'w') as json_file:
#     json_file.write(simplejson.dumps(simplejson.loads(model_json), indent=4))

print("model fitting - simplified convolutional neural network")
model.summary()
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=128)

print("saving model on disk")
model.save("./runs/simpleModel2.h5")
