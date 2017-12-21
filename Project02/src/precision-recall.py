#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import csv
from helpers import load_data_and_labels, clean_str

import os
import argparse

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

from sklearn.metrics import precision_recall_fscore_support

os.environ['KERAS_BACKEND'] = "tensorflow" # theano'

# Script parameters
parser = argparse.ArgumentParser()


parser.add_argument("-m_file", "--model_file", dest="model_file", type=str, default="paper2Model.h5", help="Name of the h5 file containing the model")
parser.add_argument("-cnn", "--cnn_algo", dest="conv_algo", type=str, default="cnn_dropout", help="Convolution algorithm; you need to choose between 'simple_conv', 'cnn_without_dropout' and 'cnn_dropout' (by default)")
parser.add_argument("-epochs", "--numb_epochs", dest="epochs", default=10, type=int, help="Number of epochs")
parser.add_argument("-pos_file", "--pos_train_file", dest="pos_file", default="preprocess_train_pos_full.txt", type=str, help="Name of the positive training file located in the data/ directory")
parser.add_argument("-neg_file", "--neg_train_file", dest="neg_file", type=str, default="preprocess_train_neg_full.txt", help="Name of the negative training file located in the data/ directory")
parser.add_argument("-emb_file", "--embeddings_file", dest="emb_file", default="glove.twitter.27B.200d.txt", type=str, help="name of the embeddings file located inside the embeddings/ folder")
parser.add_argument("-emb_dim", "--embeddings_dim", dest="emb_dim", default=200, type=int, help="Dimension of the embeddings")
parser.add_argument("-batch_size", "--batch_size", dest="batch_size", default=64, type=int, help="Size of the batches during the training")


args = parser.parse_args()

# Parameters
DATA_DIR = "./data/"
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
# Parameters
EMBEDDING_FILE = args.emb_file
EMBEDDING_DIM = args.emb_dim
VALIDATION_SPLIT = 0.04
CONV_ALGO = args.conv_algo
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size


positive_data = os.path.join(DATA_DIR, args.pos_file)
negative_data = os.path.join(DATA_DIR, args.neg_file)

print("Loading data...")
x_train, labels = load_data_and_labels(positive_data, negative_data)

# Text processing
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(x_train)
sequences = tokenizer.texts_to_sequences(x_train)

word_index = tokenizer.word_index
print("Found", len(word_index), "unique tokens.")

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

# Randomly shuffle data and split train/test data
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

print("Load model")
loaded_model = load_model(os.path.join("./runs/", args.model_file))

print("Calculation of the predictions")
print("Found", len(word_index), "unique tokens.")


# Predictions
z = loaded_model.predict(x_val, verbose=1)
prediction = np.argmax(z, axis=-1)


score, acc = loaded_model.evaluate(x_val, y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

precision, recall, _, _ = precision_recall_fscore_support(y_test, prediction, average='macro')
print("Precision", precision)
print("Recall", recall)


