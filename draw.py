import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches
import random

import seaborn as sn
import tensorflow as tf

from ipywidgets import widgets


with open(os.path.join(os.getcwd(),'categories.txt')) as f:
     categories = [line.rstrip('\n') for line in f]


"""Draws and labels a misclassified data point.

Takes a vector of sketch data, labels, and predicted labels, and draws one of
the vector entries with title as label.

Args:
  index:
    Index of which drawing to create.
  X:
    Sketches in array of drawings.
  y:
    True labels in integer array.
  y_predict:
    Predicted labels in integer array.
"""
def draw_label_probs(X, y, y_predict, category, index=None):

    draw_indices = [i for i, c in enumerate(zip(y, y_predict)) if categories[c[0]] == category]
    X_draw = [X[i] for i in draw_indices]
    y_draw = [y[i] for i in draw_indices]
    y_predict_draw = [y_predict[i] for i in draw_indices]

    if index is None: index = random.randint(0, len(draw_indices))

    palette = sn.color_palette("husl", 8)

    fig, axs = plt.subplots(1, 2, figsize=(8,4))
    for stroke in X_draw[index]:
        axs[0].plot(stroke[0], stroke[1], color='k', linewidth=3)
    true_label = categories[y_draw[index]] if isinstance(y_draw[index], int) else y_draw[index]
    axs[0].title.set_text(f'truth = {true_label}')
    axs[1].title.set_text(f'DAISY\'s prediction')
    axs[1].barh(np.array(categories)[np.argsort(y_predict_draw[index])],
                                     y_predict_draw[index][np.argsort(y_predict_draw[index])],
                color=palette)
    axs[0].axis('off')
    axs[1].tick_params(left=False)
    axs[1].set_xlim(0,1)
    axs[0].invert_yaxis()
    for y, x in enumerate(y_predict_draw[index][np.argsort(y_predict_draw[index])]):
        axs[1].annotate(f"   {x*100:.1f}%", xy=(x, y), va='center')
    fig.tight_layout()
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['bottom'].set_visible(False)
    axs[1].get_xaxis().set_ticks([])
    plt.show()


def draw_misclassification_grid(X, y, y_predict, true_category, index=None):

    in_class = [i for i, c in enumerate(zip(y, y_predict)) if categories[c[0]] == true_category]
    max_num_misclass = 5
    fig, axs = plt.subplots(1, max_num_misclass, figsize=(10,2))
    fig.suptitle(f'DAISY said {true_category} for these drawings:', color='none', size=15, weight='bold')
    for ind_drawing in range(max_num_misclass):
        axs[ind_drawing].axis('off')
        axs[ind_drawing].invert_yaxis()
        axs[ind_drawing].set_title(true_category, color='none', size=10, weight='bold', y=0.0,
                                   bbox=dict(facecolor='none', edgecolor='none'))
    fig.tight_layout()

    def select_category(false_category):
        fig.suptitle(f'DAISY says {true_category} for these drawings', color='none', size=15, weight='bold')
        draw_indices = [i for i, c in enumerate(zip(y, y_predict)) if categories[c[0]] == true_category and categories[np.argmax(c[1])] == false_category]
        X_draw = [X[i] for i in draw_indices]
        if false_category != None:
            for ind_drawing in range(max_num_misclass):
                axs[ind_drawing].set_title(true_category, color='none', size=10, weight='bold', y=0.0,
                           bbox=dict(facecolor='none', edgecolor='none'))
                for line in axs[ind_drawing].lines: line.remove()
            for ind_drawing in range(np.min([max_num_misclass, len(X_draw)])):
                pred_category = categories[np.argmax(y_predict[draw_indices][ind_drawing])]
                axs[ind_drawing].set_title(true_category, color='k', size=8, weight='bold', y=-0.1,
                                           bbox=dict(facecolor='w', edgecolor='k'))
                for stroke in X_draw[ind_drawing]:
                    axs[ind_drawing].plot(stroke[0], stroke[1], color='k')
                fig.suptitle(f'DAISY says {pred_category} for these drawings',
                             color='mediumvioletred', size=15, weight='bold')
            fig.canvas.draw_idle()

    c = widgets.Dropdown(options=[cat for cat in categories if cat != true_category], description='category:', value=None)
    widgets.interact(select_category, false_category=c)


"""Draws and labels a misclassified data point.

Takes a vector of sketch data, labels, and predicted labels, and draws one of
the vector entries with title as label.

Args:
  index:
    Index of which drawing to create.
  X:
    Sketches in array of drawings.
  y:
    True labels in integer array.
  y_predict:
    Predicted labels in integer array.
"""
def draw_misclassification(X, y, y_predict, true_category, false_category, index=None):

    draw_indices = [i for i, c in enumerate(zip(y, y_predict)) if categories[c[0]] == true_category and categories[np.argmax(c[1])] == false_category]
    X_draw = [X[i] for i in draw_indices]
    y_draw = [y[i] for i in draw_indices]
    y_predict_draw = [y_predict[i] for i in draw_indices]

    if index is None: index = random.randint(0, len(draw_indices))

    plt.figure()
    for stroke in X_draw[index]:
        plt.plot(stroke[0], stroke[1], color='k', linewidth=3)
        true_label = categories[y_draw[index]] if isinstance(y_draw[index], int) else y_draw[index]
        predicted_label = categories[np.argmax(y_predict_draw[index])]
        plt.title(f'truth = {true_label}, predicted = {predicted_label}', color='crimson', weight='bold')
    plt.axis('off')
    plt.gca().invert_yaxis()
    plt.show()


def draw_label_probs(X, y, y_predict, category, index=None):

    draw_indices = [i for i, c in enumerate(zip(y, y_predict)) if categories[c[0]] == category]
    X_draw = [X[i] for i in draw_indices]
    y_draw = [y[i] for i in draw_indices]
    y_predict_draw = [y_predict[i] for i in draw_indices]

    if index is None: index = random.randint(0, len(draw_indices))

    palette = sn.color_palette("husl", 8)

    fig, axs = plt.subplots(1, 2, figsize=(8,4))
    for stroke in X_draw[index]:
        axs[0].plot(stroke[0], stroke[1], color='k', linewidth=3)
    true_label = categories[y_draw[index]]
    axs[0].set_title(f'{true_label}', weight='bold', y=0.0, size=15, bbox=dict(facecolor='w', edgecolor='k'))
    axs[1].set_title(f'DAISY says:', weight='bold', size=15, color='mediumvioletred')
    axs[1].barh(np.array(categories)[np.argsort(y_predict_draw[index])], y_predict_draw[index][np.argsort(y_predict_draw[index])],
                color=palette)
    axs[0].axis('off')
    axs[1].tick_params(left=False)
    axs[1].set_xlim(0,1)
    axs[0].invert_yaxis()
    for y, x in enumerate(y_predict_draw[index][np.argsort(y_predict_draw[index])]):
        axs[1].annotate(f"   {x*100:.1f}%", xy=(x, y), va='center')
    fig.tight_layout()
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['bottom'].set_visible(False)
    axs[1].get_xaxis().set_ticks([])
    axs[1].xaxis.set_tick_params(size=15, color='mediumvioletred')
    plt.show()

"""Draws and labels a single raw data point.

Takes a vector of sketch data, labels, and predicted labels, and draws one of
the vector entries with title as label.

Args:
  index:
    Index of which drawing to create.
  X:
    Sketches in array of drawings.
  y:
    True labels in integer array.
  y_predict:
    Predicted labels in integer array.
"""
def draw_and_label_widget(index, test_dataset, y_predict):

    for count, data in enumerate(test_dataset):
        if count == index:
            raw_data, data, label = data

    fig, ax = plt.subplots(figsize=(5,4))

    for stroke in raw_data:
        plt.plot(stroke[0], stroke[1], color='k', linewidth=3)

    true_label = categories[label]
    predicted_label = categories[np.argmax(y_predict[index])]
    plt.title(f'{true_label}', size=20, weight='bold', y=0.0,
              bbox=dict(facecolor='w', edgecolor='k'))
    ax.axis('off')
    ax.invert_yaxis()

    def label_plot(var):
        if var:
            plt.text(0.5, -0.09, f'DAISY says: {predicted_label}', weight='bold', size=15, color='mediumvioletred', ha='center',
                     transform=ax.transAxes, bbox=dict(facecolor='w', edgecolor='none'))
        fig.canvas.draw_idle()

    widgets.interact(label_plot, var=widgets.ToggleButton(value=False, description='DAISY says...'))


def draw_and_label(index, X, y, y_predict):

    plt.figure(figsize=(5,4))
    ax = plt.gca()

    for stroke in X[index]:
        plt.plot(stroke[0], stroke[1], color='k', linewidth=3)

    true_label = categories[y[index]] if isinstance(y[index], int) else y[index]
    predicted_label = categories[np.argmax(y_predict[index])]
    plt.text(0.5, 1.0, f'DAISY says: {predicted_label}', weight='bold', size=10, transform=ax.transAxes)
    plt.title(f'{true_label}', size=20, weight='bold', y=0.0,
              bbox=dict(facecolor='w', edgecolor='k'))
    plt.axis('off')
    plt.gca().invert_yaxis()
    plt.show()


"""Draws a bunch of images from a certain category.

"""
def draw_grid_widget(data, category, n_rows=3, n_cols=8, start_index=None):

    category_data = data.filter(lambda raw_data, data, label: label == categories.index(category))

    if start_index is None: start_index = 0

    iterator = iter(category_data)
    i = 0
    while i < start_index:
        next(iterator)
        i += 1

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))
    rectangles = []
    plt.suptitle(category, weight='bold', y=0.1, size=15, bbox=dict(facecolor='none', edgecolor='k'))
    for i in range(n_rows):
        for j in range(n_cols):
            raw_data, simple_data, label = next(iterator)
            sketch_num = start_index + i * n_cols + j
            for line in raw_data:
                axs[i,j].plot(line[0], line[1], color='k', linewidth=2)
            axs[i,j].axis('off')
            axs[i,j].invert_yaxis()
            rectangles.append(patches.Rectangle((0,0), 110, 110, color="gray", alpha=0.0, transform=axs[i,j].transAxes))
            axs[i,j].add_patch(rectangles[sketch_num-start_index])

    def select_plot(row_selector, col_selector):
        if row_selector != None and col_selector != None:
            for i in range(n_rows):
                for j in range(n_cols):
                    rectangles[i * n_cols + j].set_alpha(0.0)
            rectangles[(row_selector-1) * n_cols + (col_selector-1)].set_alpha(0.3)
            fig.canvas.draw_idle()

    r = widgets.Dropdown(options=list(range(1,n_rows+1)), description='Row:', value=None)
    c = widgets.Dropdown(options=list(range(1,n_cols+1)), description='Column:', value=None)
    widgets.interact(select_plot, row_selector=r, col_selector=c)

"""Draws a bunch of images from a certain category.

"""
def draw_grid(X, y, category, n_rows=3, n_cols=8, start_index=None):

    draw_indices = [i for i, c in enumerate(y) if c == category]
    X_draw = [X[i] for i in draw_indices]

    stop = len(X_draw) - n_rows * n_cols
    if start_index is None: start_index = random.randint(0, stop)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))
    plt.suptitle(category, color='crimson', weight='bold')
    for i in range(n_rows):
        for j in range(n_cols):
            sketch_num = start_index + i * n_cols + j
            for line in X_draw[sketch_num]:
                axs[i,j].plot(line[0], line[1], color='k', linewidth=2)
            axs[i,j].axis('off')
            axs[i,j].invert_yaxis()
    plt.show()


"""Draws and labels a single raw data point.

Takes a vector of sketch data, labels, and predicted labels, and draws one of
the vector entries with title as label.

Args:
  index:
    Index of which drawing to create.
  X:
    Sketches in array of drawings.
  y:
    True labels in integer array.
"""
def draw(data, category, index=None):

    category_data = data.filter(lambda raw_data, data, label: label == categories.index(category))

    if index is None: index = 0

    for i, d in enumerate(category_data):
        if i >= index: break

    plt.figure(figsize=(5,4))
    raw_data, simple_data, label = d
    for stroke in raw_data:
        plt.plot(stroke[0], stroke[1], color='k', linewidth=3)
    plt.title(f'{categories[label]}', weight='bold', size=20, y=-0.1, bbox=dict(facecolor='w', edgecolor='k'))
    plt.axis('off')
    plt.gca().invert_yaxis()
    plt.show()
