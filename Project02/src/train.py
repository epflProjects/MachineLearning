#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from helpers import load_data_and_labels, get_embeddings
from models import get_model_simple_convolution, get_model_paper_convolution, get_model_paper_2_convolution

import os
import argparse

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

os.environ['KERAS_BACKEND'] = "tensorflow" # theano'

# Script parameters
parser = argparse.ArgumentParser()

parser.add_argument("-conv", "--convolution", dest="conv_algo", type=str, default="complex_conv", help="Convolution algorithm; you need to choose between 'simple_conv' and 'complex_conv' (by default)")
parser.add_argument("-epochs", "--numb_epochs", dest="epochs", default=10, type=int, help="Number of epochs")
parser.add_argument("-pos_file", "--pos_train_file", dest="pos_file", default="preprocess_train_pos_full.txt", type=str, help="Name of the positive training file located in the data/ directory")
parser.add_argument("-neg_file", "--neg_train_file", dest="neg_file", type=str, default="preprocess_train_neg_full.txt", help="Name of the negative training file located in the data/ directory")
parser.add_argument("-emb_file", "--embeddings_file", dest="emb_file", default="glove.twitter.27B.200d.txt", type=str, help="name of the embeddings file located inside the embeddings/ folder")
parser.add_argument("-emb_dim", "--embeddings_dim", dest="emb_dim", default=200, type=int, help="Dimension of the embeddings")
parser.add_argument("-batch_size", "--batch_size", dest="batch_size", default=64, type=int, help="Size of the batches during the training")

args = parser.parse_args()

# Parameters
DATA_DIR = "./data/"
MAX_SEQUENCE_LENGTH = 1000 # TODO how to fix?
MAX_NB_WORDS = 20000
EMBEDDING_FILE = args.emb_file
EMBEDDING_DIM = args.emb_dim
VALIDATION_SPLIT = 0.04
CONV_ALGO = args.conv_algo
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size

conv_algorithms = ["simple_conv", "complex_conv"]

positive_data = os.path.join(DATA_DIR, args.pos_file)
negative_data = os.path.join(DATA_DIR, args.neg_file)

print("Loading data...")
x_text, labels = load_data_and_labels(positive_data, negative_data)

# Text processing
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(x_text)
sequences = tokenizer.texts_to_sequences(x_text)

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

print('Number of positive and negative reviews in training and validation set')
print(y_train.sum(axis=0))
print(y_val.sum(axis=0))

print("Importing embeddings...")
embeddings_index = get_embeddings(EMBEDDING_FILE)

if CONV_ALGO not in conv_algorithms:
    raise ValueError("The convolution algorithms chosen doesn't exist; you have the choice between 'simple_conv' and 'complex_conv'")
elif CONV_ALGO == conv_algorithms[0]:
    model = get_model_simple_convolution(embeddings_index, word_index, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM)
elif CONV_ALGO == conv_algorithms[1]:
    model = get_model_paper_2_convolution(embeddings_index, word_index, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM)

print("model fitting - simplified convolutional neural network")
model.summary()
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE)

print("saving model on disk")
model.save("./runs/paper2ModelTest.h5")
