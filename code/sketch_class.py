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
from ipywidgets import widgets

from preprocess import preprocess_student_data
from draw import draw_label_probs


RANDOM_STATE = 10
with open(os.path.join(os.getcwd(),'categories.txt')) as f:
     categories = [line.rstrip('\n') for line in f]


"""Splits the data into training, testing, and validation sets.

Takes dataset and shuffles then splits according to input percentages for
testing, training, and validation sets.

Args:
  dataset:
    Full dataset, tensorflow object
  dataset_size:
    Size of full dataset, integer.
  train_split:
    Percent of samples in training set, float.
  validation_split:
    Percent of samples in validation set, float.
  test_split:
    Percent of samples in testing set, float.
  shuffle_size:
    Shuffle buffer size, integer.

Returns:

  Multiple tuples(train_dataset, validation_dataset, test_dataset), and
  (train_size, validation_size, test_size) where:

  train_dataset:
    Training dataset, tensorflow dataset.
  validation_dataset:
    Validation dataset, tensorflow dataset.
  test_dataset:
    Testing dataset, tensorflow dataset.
  train_size:
    Number of training samples, integer.
  test_size:
    Number of testing samples, integer.
  validation_size:
    Number of validation samples, integer.
"""
def get_dataset_partitions(dataset, dataset_size, train_split,
                           validation_split, test_split):

    assert (train_split + test_split + validation_split) == 1

    train_size = int(train_split * dataset_size)
    validation_size = int(validation_split * dataset_size)
    test_size = dataset_size - train_size - validation_size

    #return dataset.cache().prefetch(tf.data.AUTOTUNE)
    train_dataset = dataset.take(train_size).cache().prefetch(tf.data.AUTOTUNE)
    validation_dataset = dataset.skip(train_size).take(validation_size).cache().prefetch(tf.data.AUTOTUNE)
    test_dataset = dataset.skip(train_size).skip(validation_size).cache().prefetch(tf.data.AUTOTUNE)

    return ((train_dataset, validation_dataset, test_dataset),
            (train_size, validation_size, test_size))


"""Compiles and fits the AI model.

Architecture is defined, model is compiled, trained, and saved. Model consists
of two LSTM layers, along with dropout layers, and two fully connected layers,
as well as a softmax. Note that this model, trained using around 1k samples
per class, will give only around 85% accuracy. This works well for the purposes
of a tutorial/demo, where we want to see some incorrect examples. Tuning the
model parameters can easily produce a more accurate model. Also saves and
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
  max_n_pts:
    Number of pts in each drawing, integer.
  train_dataset:
    Training data, tensorflow dataset.
  validate_dataset:
    Validation data, tensorflow dataset.
  train_model:
    Whether or not to train the model or load it, boolean.

Returns:

  The AI model that predicts categories for preprocessed sketch drawings.
"""
def compile_and_fit_model(model_dir, model_name, epochs, batch_size, max_n_pts,
                          train_dataset, validate_dataset, train_model=True):

    if train_model:
        model = Sequential()
        model.add(Reshape((max_n_pts, 2)))
        model.add(Masking(mask_value=0.0))
        model.add(LSTM(64, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(64))
        model.add(Dropout(0.3))
        model.add(Dense((32)))
        model.add(Dropout(0.3))
        model.add(Dense((8)))
        model.add(Activation('softmax'))

        # compile and set optimizer/metrics/loss function
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='adam',
                      metrics=['sparse_categorical_accuracy'])

        # train the model
        print(f'[Training AI]')
        train_dataset_map = train_dataset.map(lambda raw_data, data, label: (data, label)).cache().prefetch(tf.data.AUTOTUNE)
        train_dataset_padded = train_dataset_map.padded_batch(batch_size, padded_shapes=([max_n_pts,2], []))
        validate_dataset_map = validate_dataset.map(lambda raw_data, data, label: (data, label)).cache().prefetch(tf.data.AUTOTUNE)
        validate_dataset_padded = validate_dataset_map.padded_batch(batch_size, padded_shapes=([max_n_pts,2], []))

        model.fit(train_dataset_padded, epochs=epochs,
                  validation_data=validate_dataset_padded)

        # save the model
        model.save(f'{model_dir}/{model_name}.h5')

    else:

        model = load_model(f'{model_dir}/{model_name}.h5')

    return model


"""Retrieves preprocessed data.

Fetches preprocessed data from files. Saves into a tensorflow dataset that
contains the raw drawing data, the simplified data for model compilation, and
the category label as an integer.

Args:
  preprocess_data_dir:
    String. Absolute directory path to preprocessed data directory.
  n_per_class:
    Integer. Number of images to use for training from each class.

Returns:

  Tensorflow dataset containing raw drawing data, simplified (preprocessed)
  data, and integer category label.

"""
def get_data(preprocess_data_dir, n_per_class):

    def gen(preprocess_data_dir):
        for category in categories:
            count = 0
            with jsonlines.open(f'{preprocess_data_dir}/preprocessed_{n_per_class}-per-class.ndjson') as reader:
                for obj in reader:
                    raw_data = tf.ragged.constant(obj['raw_data'])
                    data = tf.constant(obj['data'])
                    label = tf.constant(obj['label'])
                    yield raw_data, data, label
                    if count == (n_per_class-1): break
                    count += 1

    dataset = tf.data.Dataset.from_generator(lambda : gen(preprocess_data_dir),
                                             output_signature=(tf.RaggedTensorSpec(shape=(None, None, None), dtype=tf.float32),
                                                               tf.TensorSpec(shape=(None, 2), dtype=tf.float32),
                                                               tf.TensorSpec(shape=(), dtype=tf.int32)))

    #return dataset.cache().prefetch(tf.data.AUTOTUNE)
    return dataset


"""Preprocesses and predicts student data.

Preprocesses student data and makes predictions, widget-based.

Args:
  student_data_dir:
    String. Absolute directory path to student data directory.
  category:
    String. Category of student drawing to process.
  num:
    Integer. Number of student drawing within that category to process.

Returns:

  Tensorflow dataset containing raw drawing data, simplified (preprocessed)
  data, and integer category label.

"""
def student_predict_and_plot(student_data_dir, max_n_pts, img_size, model, batch_size):

    def select_plot(category, number):
        
        category_dir = os.path.join(student_data_dir, f'{category}')
        file_list = os.listdir(category_dir)
        valid_files = []
        
        for file in file_list:

            name = file.split('.svg')[0]
            try:
                iname = int(name)
                valid_files.append(int(file.split('.svg')[0]))
            except ValueError:
                pass

        if number in valid_files:

            student_dataset = preprocess_student_data(max_n_pts, img_size, number, category)
        
            student_dataset_map = student_dataset.map(lambda raw_data, data, label: (data, label))
            student_dataset_padded = student_dataset_map.padded_batch(batch_size, padded_shapes=([max_n_pts,2], []))
        
            y_student_pred = model.predict(student_dataset_padded)
        
            draw_label_probs(student_dataset, y_student_pred, category, index=number-1)

        else:

            print(f'[ERROR]: no file named {number}.svg in {category_dir}, please retry!')

    category = widgets.Dropdown(options=categories, description='category:', value=categories[0])
    number = widgets.BoundedIntText(min=1, value=1, description='drawing number:')

    widgets.interact(select_plot, category=category, number=number)
