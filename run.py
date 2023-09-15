import os
import numpy as np

from xml.dom import minidom
from svg.path import parse_path

from keras.models import load_model

import draw
import sketch_class
from preprocess import vector_process, student_process


IMG_SIZE = 256
RANDOM_STATE = 10

train = False
preprocess = False

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

    n_per_class = 10000

    train_percent = 0.75
    test_percent = 0.10
    validation_percent = 0.15

    epochs = 30
    batch_size = 128

    model_name = 'DAISY'

    data_dir = os.path.join(os.getcwd(), 'data')
    preprocessed_data_dir = os.path.join(data_dir, 'preprocessed')
    raw_data_dir = os.path.join(data_dir, 'raw')
    model_dir = os.path.join(os.getcwd(), 'models')

    max_n_pts, vector_data, label_data = sketch_class.get_data(raw_data_dir,
                                                               n_per_class)
    int_label_data = [categories.index(label) for label in label_data]

    splitted_data = sketch_class.split_data(train_percent, test_percent,
                                            validation_percent, vector_data,
                                            int_label_data)
    (raw_X_train, raw_X_test, raw_X_validate,
     y_train, y_test, y_validate) = splitted_data

    if preprocess:
        X_preprocessed = sketch_class.preprocess_data(max_n_pts,
                                                      preprocessed_data_dir,
                                                      raw_X_train, raw_X_test,
                                                      raw_X_validate)
        X_train, X_test, X_validate = X_preprocessed
    else:
        X_train = np.load(f'{preprocessed_data_dir}/x_train.npy')
        X_test = np.load(f'{preprocessed_data_dir}/x_test.npy')
        X_validate = np.load(f'{preprocessed_data_dir}/x_validate.npy')

    if train:
        model = sketch_class.compile_and_fit_model(model_dir, model_name,
                                                   epochs, batch_size, X_train,
                                                   np.array(y_train),
                                                   X_validate,
                                                   np.array(y_validate))
    else:
        model = load_model(f'{model_dir}/{model_name}.h5')

    y_predict = model.predict(np.array(X_test)) # predict on test data

    # plot some of the test results
    for count, vector in enumerate(raw_X_test[:5]):
        draw.draw_and_label(count, raw_X_test, y_test, y_predict)

    draw.print_and_plot_confusion(y_test, y_predict)

    # processing sample student drawing
    #svg_dom = minidom.parse('./data/student_svgs/sample.svg')
    svg_dom = minidom.parse('./data/student_svgs/1.svg')

    path_strings = [path.getAttribute('d') for path in svg_dom.getElementsByTagName('path')]
    tmat_strings = [path.getAttribute('transform') for path in svg_dom.getElementsByTagName('path')]

    X_full = []
    count = 0
    for path_string, tmat_string in zip(path_strings, tmat_strings):
        X_full.append([])
        x_full = []
        y_full = []
        path_data = parse_path(path_string)

        points = [p.start for p in path_data[1:]]
        x = [ele.real for ele in points]
        y = [ele.imag for ele in points]

        temp = tmat_string.split('(')[1]
        mat_list = np.array([float(x) for x in (temp.split(')')[0]).split(',')])
        tmat = np.append(mat_list.reshape(2,3, order='F'), [[0,0,1]], axis=0)

        x_trans = []
        y_trans = []
        for xx, yy in zip(x, y):
            stack = np.array([[xx], [yy], [1]])
            stack_trans = tmat.dot(stack)
            x_trans.append(float(stack_trans[0]))
            y_trans.append(float(stack_trans[1]))

        x_full = x_full + x_trans
        y_full = y_full + y_trans

        X_full[count].append(x_full)
        X_full[count].append(y_full)
        count = count + 1

    svg_pts = [X_full]

    y_test = [1]
    X_student = np.array(vector_process(max_n_pts, IMG_SIZE, svg_pts))
    y_student = model.predict(np.array(X_student))
    print(y_student)
    draw.draw_and_label(0, svg_pts, y_test, y_student)


    y_student, X_student, svg_pts = student_process(max_n_pts, IMG_SIZE, 1, 'hurricane')
    y_student_pred = model.predict(np.array(X_student))

    #draw.draw_and_label(0, svg_pts, y_student_pred, y_student)
    draw.draw_and_label(0, svg_pts, y_student, y_student_pred)


if __name__ == '__main__':
    main()
