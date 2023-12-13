import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import random

import pandas as pd
import seaborn as sn
import tensorflow as tf

from ipywidgets import widgets


IMG_SIZE = 256
RANDOM_STATE = 10

train = True
preprocess = True

with open(os.path.join(os.getcwd(),'categories.txt')) as f:
     categories = [line.rstrip('\n') for line in f]


def print_grades(test_dataset, y_predict, category_to_grade=None):

    grade_cutoffs = [0.925, 0.9, 0.875, 0.825, 0.8, 0.775, 0.725, 0.7, 0.675, 0.625, 0.6, 0.0]
    grade_letters = ['A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D+', 'D', 'D', 'F']

    def grade_convert(score):
        count = 0
        for cutoff in grade_cutoffs:
            if score > cutoff: return grade_letters[count]
            else: count = count + 1

    max_cat_len = np.max([len(cat) for cat in categories])
    
    num_correct = [0]*(len(categories)+1)
    num_total = [0]*(len(categories)+1)
    for count, d in enumerate(test_dataset):
        raw_test_data, data, true_label = d
        pred_label = np.argmax(y_predict[count])
        if true_label == pred_label:
            num_correct[true_label] += 1
            num_correct[-1] += 1
        num_total[true_label] += 1
        num_total[-1] += 1

    grades = []
    for g in zip(num_correct, num_total):
        nc, nt = g
        grades.append(nc / nt)

    s = ' '
    if category_to_grade == 'total':
        print(f'TOTAL {s*(max_cat_len-len(category_to_grade))} test grade :  {grades[-1]*100:0.2f}%')
        print(f'{s*(max_cat_len+15)} {grade_convert(grades[-1])}')
    elif category_to_grade == None:
        for category_to_grade in categories:
            print(f'{category_to_grade.upper()} {s*(max_cat_len-len(category_to_grade))} test grade : {grades[categories.index(category_to_grade)]*100:0.2f}%')
            print(f'{s*(max_cat_len+14)} {grade_convert(grades[categories.index(category_to_grade)])}')
            print()
    else:
        print(f'{category_to_grade.upper()} {s*(max_cat_len-len(category_to_grade))} test grade : {grades[categories.index(category_to_grade)]*100:0.2f}%')
        print(f'{s*(max_cat_len+14)} {grade_convert(grades[categories.index(category_to_grade)])}')


def category_report(report_category, test_dataset, y_predict):

    report_index = categories.index(report_category)

    num_incorrect = [0]*(len(categories))
    for count, d in enumerate(test_dataset):
        raw_test_data, data, true_label = d
        pred_label = np.argmax(y_predict[count])
        if true_label == categories.index(report_category):
            num_incorrect[pred_label] += 1

    s = ' '
    print('number of...')
    for category in categories:
        if report_category != category:
            spaces_len = np.max([len(category) for category in categories]) - len(category)
            category_index = categories.index(category)
            print(f'    {report_category.upper()} drawings DAISY thought were {category.upper()} drawings: {s*spaces_len}{num_incorrect[category_index]}')


def run_test(test_dataset, batch_size, max_pts, model):
    
    test_dataset_map = test_dataset.map(lambda raw_data, data, label: (data, label))
    test_dataset_padded = test_dataset_map.padded_batch(batch_size, padded_shapes=([max_pts,2], [])).cache().prefetch(tf.data.AUTOTUNE)

    return model.predict(test_dataset_padded)
