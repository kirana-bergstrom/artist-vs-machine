import os

import numpy as np
import ipywidgets as widgets

import code.draw as draw
import code.utils as utils
import code.test as test
import code.sketch_class as sketch_class
import code.preprocess as preprocess


IMG_SIZE = 256

to_preprocess = False
to_train = False

with open(os.path.join(os.getcwd(),'categories.txt')) as f:
     categories = [line.rstrip('\n') for line in f]


"""Creates, runs an AI for classification of Quick, Draw! sketches.

The purpose of this module is to create and run a basic AI that identifies
simple sketches from a few categories of the Google Quick, Draw! dataset. The
module is purposely simplified so that it can be used in a tutorial aimed at
middle to high school level students where the students will submit their own
sketches mid-way through the tutorial, and designed so that the model will
demonstrate several introductory AI topics and concepts.
"""
def main():

    RANDOM_STATE = 10
    MAX_N_PTS = 196
    NUM_PREPROCESSED = 1500
    NUM_DATA = 1500

    train_split = 0.70
    test_split = 0.15
    validation_split = 0.15

    batch_size = 64
    epochs = 10

    AI_name = 'DAISY'

    data_dir = os.path.join(os.getcwd(), 'data')
    preprocessed_data_dir = os.path.join(data_dir, 'preprocessed')
    raw_data_dir = os.path.join(data_dir, 'raw')
    student_data_dir = os.path.join(data_dir, 'student')

    model_dir = os.path.join(os.getcwd(), 'models')

    preprocess.preprocess_raw_data(MAX_N_PTS, IMG_SIZE, raw_data_dir,
                                   preprocessed_data_dir, NUM_PREPROCESSED,
                                   preprocess=to_preprocess,
                                   random_seed=RANDOM_STATE)

    dataset = sketch_class.get_data(preprocessed_data_dir, NUM_DATA)

    datasets, sizes = sketch_class.get_dataset_partitions(dataset,
                                                          NUM_DATA*len(categories),
                                                          train_split=train_split,
                                                          test_split=test_split,
                                                          validation_split=validation_split)
    train_dataset, validate_dataset, test_dataset = datasets
    train_size, val_size, test_size = sizes

    DAISY = sketch_class.compile_and_fit_model(model_dir, AI_name, epochs,
                                               batch_size, MAX_N_PTS,
                                               train_dataset, validate_dataset,
                                               train_model=to_train)

    predict_labels = test.run_test(test_dataset, batch_size, MAX_N_PTS, DAISY)

    test.print_grades(test_dataset, predict_labels, 'total')
    test.print_grades(test_dataset, predict_labels)

    test.category_report('hurricane', test_dataset, predict_labels)
    test.category_report('lightning', test_dataset, predict_labels)

if __name__ == '__main__':
    main()
