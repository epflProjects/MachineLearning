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

os.environ['KERAS_BACKEND'] = "tensorflow" # theano'

# Script parameters
parser = argparse.ArgumentParser()

parser.add_argument("-m_file", "--model_file", dest="model_file", type=str, default="cnnModel.h5", help="Name of the h5 file containing the model. By default: cnnModel.h5.")
parser.add_argument("-pos_file", "--pos_train_file", dest="pos_file", default="preprocess_train_pos_full.txt", type=str, help="Name of the positive training file located in the data/ directory. By default: preprocess_train_pos_full.txt.")
parser.add_argument("-neg_file", "--neg_train_file", dest="neg_file", type=str, default="preprocess_train_neg_full.txt", help="Name of the negative training file located in the data/ directory. By default: preprocess_train_neg_full.txt.")
parser.add_argument("-test_file", "--test_file", dest="test_file", type=str, default="test_data.txt", help="Name of the file that you want to make predictions located in the data/ directory. By default: test_data.txt.")

args = parser.parse_args()

# Parameters
DATA_DIR = "./data/"
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000

positive_data = os.path.join(DATA_DIR, args.pos_file)
negative_data = os.path.join(DATA_DIR, args.neg_file)
test_data = os.path.join(DATA_DIR, args.test_file)

print("Loading data...")
x_train, labels = load_data_and_labels(positive_data, negative_data)

source_test = open(test_data, "r")
# clean the test_data
tests = list()
for r in source_test:
    r = r.partition(",")[2]
    tests.append(r)

tests = [s.strip() for s in tests]
x_test = [clean_str(sent) for sent in tests]

# Text processing
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(x_train)
sequences = tokenizer.texts_to_sequences(x_test)

word_index = tokenizer.word_index

print("Loading model from ", args.model_file, "file...")
loaded_model = load_model(os.path.join("./runs/", args.model_file))

print("Calculation of the predictions")
print("Found", len(word_index), "unique tokens.")

x_test = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

# Predictions
z = loaded_model.predict(x_test, verbose=1)
prediction = np.argmax(z, axis=-1)

print("rendering of the predictions.csv inside the data/ directory")
with open(os.path.join(DATA_DIR, "predictions.csv"), "w") as prediction_file:
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
