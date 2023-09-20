import os
import numpy as np
import matplotlib.pyplot as plt
import random

import jsonlines
from keras.layers import Activation, Dropout, Dense, LSTM, Reshape, Masking
from keras.models import Sequential, load_model
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
                    raw_X_train, raw_X_test, raw_X_validate,
                    y_train, y_test, y_validate):

    def gen(file):
        with jsonlines.open(file) as reader:
            for obj in reader:
                data = tf.constant(obj['simple_data'])
                label = tf.constant(obj['label'])
                yield data, label
    
    def get_dataset(file):
        generator = lambda: gen(file)
        return tf.data.Dataset.from_generator(generator,
                                              output_signature=(tf.TensorSpec(shape=(None, 2), dtype=tf.float32),
                                                                tf.TensorSpec(shape=(), dtype=tf.string)))
        
    dataset_train = get_dataset(f'data/train_data.ndjson')
    dataset_test = get_dataset(f'data/test_data.ndjson')
    dataset_validate = get_dataset(f'data/validate_data.ndjson')

    print(f'[Preprocessing training data]')
    train_dataset = vector_process(max_n_pts, IMG_SIZE, raw_X_train, y_train)
    print(f'[Preprocessing testing data]')
    test_dataset = vector_process(max_n_pts, IMG_SIZE, raw_X_test, y_test)
    print(f'[Preprocessing validation data]')
    validate_dataset = vector_process(max_n_pts, IMG_SIZE, raw_X_validate, y_validate)

    with jsonlines.open(f'{preprocessed_data_dir}/x_train.ndjson', 'w') as writer:
        writer.write_all(train_dataset)
    with jsonlines.open(f'{preprocessed_data_dir}/x_test.ndjson', 'w') as writer:
        writer.write_all(test_dataset)
    with jsonlines.open(f'{preprocessed_data_dir}/x_validate.ndjson', 'w') as writer:
        writer.write_all(validate_dataset)

    return train_dataset, test_dataset, validate_dataset


def get_dataset_partitions(ds, ds_size, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=8000):
    assert (train_split + test_split + val_split) == 1

    if shuffle:
        # Specify seed to always have the same split distribution between runs
        ds = ds.shuffle(shuffle_size, seed=10, reshuffle_each_iteration=False)

    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)

    train_ds = ds.take(train_size)    
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)

    return train_ds, val_ds, test_ds, train_size, val_size, ds_size - train_size - val_size


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
                          train_dataset, validate_dataset, train_model=True):

    if train_model:
        model = Sequential()
        model.add(Reshape((200, 2)))
        model.add(Masking(mask_value=0.0))
        model.add(LSTM(64, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(64, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(32))
        model.add(Dropout(0.2))
        model.add(Dense((8)))
        model.add(Activation('softmax'))

        # compile and set optimizer/metrics/loss function
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['sparse_categorical_accuracy'])

        # train the model
        print(f'[Training AI]')
        train_dataset_map = train_dataset.map(lambda raw_data, data, label: (data, label))
        train_dataset_padded = train_dataset_map.padded_batch(batch_size, padded_shapes=([200,2], []))
        validate_dataset_map = validate_dataset.map(lambda raw_data, data, label: (data, label))
        validate_dataset_padded = validate_dataset_map.padded_batch(batch_size, padded_shapes=([200,2], []))
        model.fit(train_dataset_padded, epochs=epochs,
                  validation_data=validate_dataset_padded)

        # save the model
        model.save(f'{model_dir}/{model_name}.h5')

    else:

        model = load_model(f'{model_dir}/{model_name}.h5')

    return model

def compile_and_fit_model2(model_dir, model_name, epochs, batch_size,
                          train_dataset, validate_dataset):

    model = Sequential()
    model.add(Reshape((200, 2)))
    model.add(tf.keras.layers.Masking(mask_value=0.0))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    model.add(Dense((8)))
    model.add(Activation('softmax'))

    # compile and set optimizer/metrics/loss function
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['sparse_categorical_accuracy'])

    # train the model
    print(f'[Training AI]')
    train_padded = train_dataset.batch(batch_size, padded_shapes=([200,2], []))
    model.fit(train_dataset, epochs=epochs,
              validation_data=validate_dataset)

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
"""

def get_data(preprocess_data_dir, n_per_class):

    def gen(preprocess_data_dir):
        for category in categories:
            count = 0
            with jsonlines.open(f'{preprocess_data_dir}/{category}.ndjson') as reader:
                for obj in reader:
                    raw_data = tf.ragged.constant(obj['raw_data'])
                    data = tf.constant(obj['data'])
                    #label = tf.constant(categories.index(obj['label']))
                    label = tf.constant(obj['label'])
                    yield raw_data, data, label
                    if count == (n_per_class-1): break
                    count += 1
    
    return tf.data.Dataset.from_generator(lambda : gen(preprocess_data_dir),
                                          output_signature=(tf.RaggedTensorSpec(shape=(None, None, None), dtype=tf.float32),
                                                            tf.TensorSpec(shape=(None, 2), dtype=tf.float32),
                                                            tf.TensorSpec(shape=(), dtype=tf.int32)))

def new_get_data(preprocess_data_dir, n_per_class):

    def gen(preprocess_data_dir):
        for category in categories:
            count = 0
            with jsonlines.open(f'{preprocess_data_dir}/{category}.ndjson') as reader:
                for obj in reader:
                    raw_data = tf.ragged.constant(obj['raw_data'])
                    data = tf.constant(obj['data'])
                    label = tf.constant(obj['label'])
                    yield raw_data, data, label
                if count == (n_per_class-1): break
                count += 1
    
    return tf.data.Dataset.from_generator(lambda : gen(preprocess_data_dir),
                                          output_signature=(tf.RaggedTensorSpec(shape=(1, None, None), dtype=tf.float32),
                                                            tf.TensorSpec(shape=(None, 2), dtype=tf.float32),
                                                            tf.TensorSpec(shape=(), dtype=tf.int32)))
