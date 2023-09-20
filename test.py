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

def print_grades(test_labels, y_predict, category_to_grade):
    
    is_correct = np.where(test_labels == np.argmax(y_predict, axis=1))[0]
    num_correct = len(is_correct)
    num_total = len(test_labels)
    num_incorrect = num_total - num_correct
    grades = [[num_correct / num_total, num_total, num_correct, num_incorrect]]

    grade_cutoffs = [0.925, 0.9, 0.875, 0.825, 0.8, 0.775, 0.725, 0.7, 0.675, 0.625, 0.6, 0.0]
    grade_letters = ['A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D+', 'D', 'D', 'F']

    def grade_convert(score):
        count = 0
        for cutoff in grade_cutoffs:
            if score > cutoff: return grade_letters[count]
            else: count = count + 1

    max_cat_len = np.max([len(cat) for cat in categories])
    for ind, category in enumerate(categories):
        in_category = np.where(test_labels == ind)[0]
        is_correct_category = np.where(test_labels[in_category] == np.argmax(y_predict[in_category], axis=1))[0]
        num_correct = len(is_correct_category)
        num_total = len(in_category)
        num_incorrect = num_total - num_correct
        grades.append([num_correct / num_total, num_total, num_correct, num_incorrect])
    
    s = ' '
    if category_to_grade == 'total':
        print(f'TOTAL {s*(max_cat_len-len(category_to_grade))} test grade :  {grades[0][0]*100:0.2f}%')
        print(f'{s*(max_cat_len+15)} {grade_convert(grades[0][0])}')
    else:
        print(f'{category_to_grade.upper()} {s*(max_cat_len-len(category_to_grade))} test grade :  {grades[categories.index(category_to_grade)+1][0]*100:0.2f}%')
        print(f'{s*(max_cat_len+15)} {grade_convert(grades[categories.index(category_to_grade)+1][0])}')
    print()


def category_report(report_category, test_labels, y_predict):

    report_index = categories.index(report_category)
    in_category = [i == report_index for i in test_labels]
    num_correct = np.sum(in_category & (test_labels == np.argmax(y_predict, axis=1)))
    num_total = np.sum(in_category)
    num_incorrect = num_total - num_correct

    s = ' '
    max_cat_len = np.max([len(cat) for cat in categories])

    print('number of...')
    for category in categories:
        if report_category != category:
            cat_index = categories.index(category)
            num = np.sum(in_category & (cat_index == np.argmax(y_predict, axis=1)))
            print(f'    {category.upper()} drawings {s*(max_cat_len-len(category))} DAISY thought were {report_category.upper()} drawings: {num}')
