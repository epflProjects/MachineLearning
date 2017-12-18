import numpy as np
import pandas as pd
import json
import simplejson
from collections import defaultdict
import re

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
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 200
VALIDATION_SPLIT = 0.2 # TODO try 0.5?

PREDICT = False

def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]

print("Loading data...")
x_text, labels = load_data_and_labels("./data/preprocess_train_pos_full.txt", "./data/preproceess_train_neg_full.txt")

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(x_text)
sequences = tokenizer.texts_to_sequences(x_text)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

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

GLOVE_DIR = "./embeddings/"
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.twitter.27B.200d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Total %s word vectors in embeddings file' % len(embeddings_index))

## SIMPLER CONVULATION: 128 filters with size 5; max pooling of 5 and 25

# embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
# for word, i in word_index.items():
#     embedding_vector = embeddings_index.get(word)
#     if embedding_vector is not None:
#         # words not found in embedding index will be all-zeros.
#         embedding_matrix[i] = embedding_vector
#
# embedding_layer = Embedding(len(word_index) + 1,
#                             EMBEDDING_DIM,
#                             weights=[embedding_matrix],
#                             input_length=MAX_SEQUENCE_LENGTH,
#                             trainable=True)
#
# sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
# embedded_sequences = embedding_layer(sequence_input)
# l_cov1 = Conv1D(128, 5, activation='relu')(embedded_sequences)
# l_pool1 = MaxPooling1D(5)(l_cov1)
# l_cov2 = Conv1D(128, 5, activation='relu')(l_pool1)
# l_pool2 = MaxPooling1D(5)(l_cov2)
# l_cov3 = Conv1D(128, 5, activation='relu')(l_pool2)
# l_pool3 = MaxPooling1D(35)(l_cov3)  # global max pooling
# l_flat = Flatten()(l_pool3)
# l_dense = Dense(128, activation='relu')(l_flat)
# preds = Dense(2, activation='softmax')(l_dense)
#
# model = Model(sequence_input, preds)
# model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
# print("Saving the model to disk in a JSON format")
# model_json = model.to_json()
# with open('data/convModel.json', 'w') as json_file:
#     json_file.write(simplejson.dumps(simplejson.loads(model_json), indent=4))
#
# print("model fitting - simplified convolutional neural network")
# model.summary()
# model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=128)
# model.save("./runs/simpleModel2.h5")


## Paper Convolution

embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)

# applying a more complex convolutional approach
convs = []
filter_sizes = [3, 4, 5]

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)

for fsz in filter_sizes:
    l_conv = Conv1D(nb_filter=128, filter_length=fsz, activation='relu')(embedded_sequences)
    l_pool = MaxPooling1D(5)(l_conv)
    convs.append(l_pool)

l_merge = Merge(mode='concat', concat_axis=1)(convs)
l_cov1= Conv1D(activation='relu', filters=128, kernel_size=5)(l_merge)
l_pool1 = MaxPooling1D(5)(l_cov1)
l_cov2 = Conv1D( activation='relu', filters=128, kernel_size=5)(l_pool1)
l_pool2 = MaxPooling1D(30)(l_cov2)
l_flat = Flatten()(l_pool2)
l_dense = Dense(128, activation='relu')(l_flat)
preds = Dense(2, activation='softmax')(l_dense)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
print("model fitting - more complex convolutional neural network")
model.summary()
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=20, batch_size=50)
print("Saving the model to disk")
model.save("./runs/complexModel.h5")

# if not PREDICT:
#     print("model fitting - more complex convolutional neural network")
#     model.summary()
#     model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=20, batch_size=50)
#     # print("Saving the model to disk in a JSON format")
#     # model_json = model.to_json()
#     # with open('data/convModel.json', 'w') as json_file:
#     #     json_file.write(simplejson.dumps(simplejson.loads(model_json), indent=4))
#     print("Saving the model to disk")
#     model.save("./runs/complexModel.h5")
#
#     ## TRY #####################################################################
#
#     source_test = open("./data/test_data.txt", "r")
#     #rdr_test = csv.reader(source_test)
#
#     tests = list()
#     for r in source_test:
#         r = r.partition(",")[2]
#         tests.append(r)
#
#     tests = [s.strip() for s in tests]
#     x_test = [clean_str(sent) for sent in tests]
#     tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
#     tokenizer.fit_on_texts(x_test)
#     sequences = tokenizer.texts_to_sequences(x_test)
#
#     word_index = tokenizer.word_index
#     print('Found %s unique tokens.' % len(word_index))
#
#     data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
#
#     indices = np.arange(data.shape[0])
#     np.random.shuffle(indices)
#     data = data[indices]
#     nb_validation_samples = int(1 * data.shape[0])
#
#     x_test = data
#     z = model.predict(x_test, verbose=1)
#     prediction = np.argmax(z, axis=-1)
#     with open("data/predictions.csv", "w") as prediction_file:
#         wtr = csv.writer(prediction_file)
#         wtr.writerow(("Id", "Prediction"))
#         line = 1
#         for pred in prediction:
#             if pred == 0:
#                 pred = str(-1)
#             else:
#                 pred = str(1)
#             wtr.writerow((line, pred))
#             line += 1
#
#     ## END TRY #####################################################################
# else:
#     weights = model.load_weights("./runs/simpleModel.h5")
#     x_test, y = load_data_and_labels("./data/test_data.txt", "./data/test_data.txt")
#     z = model.predict(x_test, batch_size=128, verbose=1)
