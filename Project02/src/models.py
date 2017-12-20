#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np

import os

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, Concatenate
from keras.models import Model

os.environ['KERAS_BACKEND']= "tensorflow"

def get_model_simple_convolution(embeddings_index, word_index, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM):
    """
    Simple Convolutional Neural Network using 128 filters of size 5 and max pooling of size 5 and global max pooling (size 35).
    Based on https://richliao.github.io/supervised/classification/2016/11/26/textclassifier-convolutional/
    Returns the compiled model.
    """
    embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    embedding_layer = Embedding(len(word_index) + 1, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=True)

    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    conv_layer = Conv1D(128, 5, activation='relu')(embedded_sequences)
    maxPool_layer = MaxPooling1D(5)(conv_layer)
    conv_layer = Conv1D(128, 5, activation='relu')(maxPool_layer)
    maxPool_layer = MaxPooling1D(5)(conv_layer)
    conv_layer = Conv1D(128, 5, activation='relu')(maxPool_layer)
    globalPool_layer = MaxPooling1D(35)(conv_layer)  # global max pooling
    flat_layer = Flatten()(globalPool_layer)
    dense_layer = Dense(128, activation='relu')(flat_layer)
    preds = Dense(2, activation='softmax')(dense_layer)

    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])

    return model

def get_model_paper_convolution(embeddings_index, word_index, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM):
    """
    """
    embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    embedding_layer = Embedding(len(word_index) + 1, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=True)

    convs = []
    filter_sizes = [3, 4, 5]

    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    for fsz in filter_sizes:
        l_conv = Conv1D(filters=128, activation='relu', filter_length=fsz)(embedded_sequences)
        l_pool = MaxPooling1D(5)(l_conv)
        convs.append(l_pool)

    l_merge = Merge(mode='concat', concat_axis=1)(convs)
    l_cov1= Conv1D(filters=128, activation='relu', kernel_size=5)(l_merge)
    l_pool1 = MaxPooling1D(5)(l_cov1)
    l_cov2 = Conv1D(filters=128, activation='relu', kernel_size=5)(l_pool1)
    l_pool2 = MaxPooling1D(30)(l_cov2)
    l_flat = Flatten()(l_pool2)
    l_dense = Dense(128, activation='relu')(l_flat)
    preds = Dense(2, activation='softmax')(l_dense)

    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])

    return model

def get_model_paper_2_convolution(embeddings_index, word_index, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM):
    """
    Convolutional Neural Network with filters of size 3, 4 and 5 and dropout probabilities of (0.5, 0.8).
    Based on http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/
    Returns the compiled model.
    """
    dropout_prob = [0.5, 0.8]
    convs = []
    filter_sizes = [3, 4, 5]

    embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype="int32")
    embedding_layer = Embedding(len(word_index) + 1, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH)(sequence_input)
    dropout_layer = Dropout(dropout_prob[0])(embedding_layer)

    for fsz in filter_sizes:
        conv_layer = Conv1D(filters=10, kernel_size=fsz, padding="valid", activation="relu", strides=1)(dropout_layer)
        conv_layer = MaxPooling1D(pool_size=2)(conv_layer)
        conv_layer = Flatten()(conv_layer)
        convs.append(conv_layer)

    layer = Concatenate()(convs) if len(convs) > 1 else convs[0]
    dropout_layer = Dropout(dropout_prob[1])(layer)
    layer = Dense(50, activation="relu")(dropout_layer)
    model_output = Dense(2, activation="sigmoid")(layer)

    model = Model(sequence_input, model_output)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])

    return model
