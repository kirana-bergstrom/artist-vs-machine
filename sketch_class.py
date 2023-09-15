import os
import numpy as np
import matplotlib.pyplot as plt
import random

import jsonlines
from keras.layers import Activation, Dropout, Dense, LSTM
from keras.models import Sequential
import pandas as pd
import seaborn as sn
from sklearn.model_selection import train_test_split
import tensorflow as tf

from preprocess import vector_process


IMG_SIZE = 256
RANDOM_STATE = 10

train = True
preprocess = True

with open(os.path.join(os.getcwd(),'categories.txt')) as f:
     categories = [line.rstrip('\n') for line in f]


"""Preprocesses raw Quick, Draw! drawing data

Takes training, testing, and validation raw drawing data and runs through a
preprocessing function, saves preprocessed data in files and returns.

Args:
  max_n_pts:
    Maximum number of coordinate points in a single drawing.
  preprocessed_data_dir:
    Directory where preprocessed data will be stored.
  raw_X_train:
    A list of x, y coordinates of training drawings.
  raw_X_test:
    A list of x, y coordinates of testing drawings.
  raw_X_validate:
    A list of x, y coordinates of validation drawings.

Returns:

  A tuple (X_train, X_test, X_validate), where

  X_train:
    An array of preprocessed training data.
  X_test:
    An array of preprocessed testing data.
  X_validate:
    An array of preprocessed validation data.
"""
def preprocess_data(max_n_pts, preprocessed_data_dir,
                    raw_X_train, raw_X_test, raw_X_validate):

    print(f'[Preprocessing training data]')
    X_train = np.array(vector_process(max_n_pts, IMG_SIZE, raw_X_train))
    print(f'[Preprocessing testing data]')
    X_test = np.array(vector_process(max_n_pts, IMG_SIZE, raw_X_test))
    print(f'[Preprocessing validation data]')
    X_validate = np.array(vector_process(max_n_pts, IMG_SIZE, raw_X_validate))

    np.save(f'{preprocessed_data_dir}/x_train.npy', X_train)
    np.save(f'{preprocessed_data_dir}/x_test.npy', X_test)
    np.save(f'{preprocessed_data_dir}/x_validate.npy', X_validate)

    return X_train, X_test, X_validate


"""Compiles and fits the AI model.

Architecture is defined, model is compiled, trained, and saved. Model consists
of three LSTM layers, along with dropout layers, and a fully connected layer,
as well as a softmax. Note that this model, trained using around 10k samples
per class, will give only around 80% accuracy. This works well for the purposes
of a tutorial/demo, where we want to see some incorrect examples. Saves and
returns the model.

Args:
  model_dir:
    Model directory to save model in, string.
  model_name:
    Model name to save model as, string.
  epochs:
    Number of samples to grab per each class, integer.
  batch_size:
    Number of samples to grab per each class, integer.
  X_train:
    Preprocessed drawing training data, array of x,y coordinates.
  y_train:
    Integer category labels corresponding to X_train, array of ints.
  X_validate:
    Preprocessed drawing validation data, array of x,y coordinates.
  y_validate:
    Integer category labels corresponding to X_validate, array of ints.

Returns:

  The AI model that predicts categories for preprocessed sketch drawings.
"""
def compile_and_fit_model(model_dir, model_name, epochs, batch_size,
                          X_train, y_train,
                          X_validate, y_validate):

    # define the NN model and layers
    model = Sequential()
    model.add(LSTM(IMG_SIZE, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(IMG_SIZE, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(IMG_SIZE))
    model.add(Dropout(0.2))
    model.add(Dense(len(categories)))
    model.add(Activation('softmax'))

    # compile and set optimizer/metrics/loss function
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['sparse_categorical_accuracy'])

    # train the model
    print(f'[Training AI]')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
              validation_data=(X_validate, y_validate))

    # save the model
    model.save(f'{model_dir}/{model_name}.h5')

    return model


"""Splits up raw data into raw testing, training, and validation sets.

Takes raw data and splits it into testing, training, and validation sets. Uses
multiple scipy split functions to get appropriate percentages.

Args:
  train_percent:
    Fraction of data to use for training.
  test_percent:
    Fraction of data to use for testing.
  validation_percent:
    Fraction of data to use for validation.
  X_data:
    A list of x, y coordinates of drawings.
  y_data:
    A list of integer category labels for X_data.

Returns:

  A tuple (X_train, X_test, X_validate, y_train, y_test, y_validate), where

  X_train:
    An array of raw training data.
  X_test:
    An array of raw testing data.
  X_validate:
    An array of raw validation data.
  y_train:
    An array of integer labels corresponding to X_train.
  y_test:
    An array of integer labels corresponding to X_test.
  y_validate:
    An array of integer labels corresponding to X_validate.
"""
def split_data(train_percent, test_percent, validation_percent,
               X_data, y_data):

    percent_split = test_percent / (validation_percent + test_percent)

    # temp is for combined test and validation
    train_splitted_data = train_test_split(X_data, y_data,
                                           test_size=test_percent,
                                           random_state=RANDOM_STATE)
    X_train, X_temp, y_train, y_temp = train_splitted_data
    splitted_data = train_test_split(X_temp, y_temp, test_size=percent_split,
                                     random_state=RANDOM_STATE+33)
    X_validate, X_test, y_validate, y_test = splitted_data

    return X_train, X_test, X_validate, y_train, y_test, y_validate


"""Gets raw drawing data in x, y coordinate list from Quick, Draw! ndjson.

Gets a list of what we call raw drawing data, which is actually preprocessed
by taking just the labels and the x, y coordinates of each stroke, from the
Quick, Draw! Google dataset raw files. Also computes the max number of points
in each drawing (totaled over each stroke).

Args:
  raw_data_dir:
    Data directory where raw Quick, Draw! data is stored, string.
  n_per_class:
    Number of samples to grab per each class, integer.

Returns:

  A tuple (max_n_pts, vector_data, label_data), where

  max_n_pts:
    Maximum number of points in any single drawing, integer.
  vector_data:
    List of raw drawing data.
  label_data:
    List of strings of labels corresponding to vector_data.
"""
def get_data(raw_data_dir, n_per_class):

    max_n_pts = 0

    vector_data = []
    label_data = []
    for cat in categories:
        count = 0
        with jsonlines.open(f'{raw_data_dir}/{cat}.ndjson') as reader:
            for obj in reader:
                vector_data.append(obj['drawing'])
                label_data.append(cat)
                n_pts = 0
                for stroke in obj['drawing']:
                    n_pts += len(np.array(stroke).T)
                if  n_pts > max_n_pts: max_n_pts = n_pts
                if count == (n_per_class-1): break
                count += 1

    return max_n_pts, vector_data, label_data
