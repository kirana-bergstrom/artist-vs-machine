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


"""Draws and labels a grid of misclassified drawings.

Draws up to 5 misclassified drawings for each category corresponding to a
true category. For example, if the true category is lightning, then
will draw up to 5 drawings that were actually lightning but were misclassified
by DAISY as clouds, up to 5 that were actually lightning but were misclassified
by DAISY as tornados, etc.

Args:
  test_dataset:
    Test dataset (tf dataset) including labels and vector data.
  y_predict:
    DAISY's predictions for the true dataset (np array).
  true_category:
    True category, string.
  index:
    Index (within category) to start on.
"""
def draw_misclassification_grid(test_dataset, y_predict, true_category, index=None):

    plt.close()

    if index == None: index = 0

    max_num_misclass = 5
    fig, axs = plt.subplots(1, max_num_misclass, figsize=(10,2))
    fig.suptitle(f'DAISY says: {true_category}', color='none', size=15, weight='bold')
    for ind_drawing in range(max_num_misclass):
        axs[ind_drawing].set_xticks([])
        axs[ind_drawing].set_yticks([])
        axs[ind_drawing].spines['bottom'].set_color('w')
        axs[ind_drawing].spines['top'].set_color('w') 
        axs[ind_drawing].spines['right'].set_color('w')
        axs[ind_drawing].spines['left'].set_color('w')
        axs[ind_drawing].invert_yaxis()
        axs[ind_drawing].set_title(true_category, color='none', size=10, weight='bold', y=0.0,
                                   bbox=dict(facecolor='none', edgecolor='none'))
    fig.tight_layout()

    def select_category(false_category):
        fig.suptitle(f'DAISY says: {true_category}', color='none', size=15, weight='bold')
        
        if false_category != None:
            X_draw = []
            y_draw = []
            for_index = 0
            for count, d in enumerate(test_dataset):
                raw_data, data, true_label = d
                if true_label == categories.index(true_category) and np.argmax(y_predict[count]) == categories.index(false_category):
                    if for_index >= index:
                        X_draw.append(raw_data)
                        y_draw.append(y_predict[count])
                    for_index += 1
                    
            for ind_drawing in range(max_num_misclass):
                axs[ind_drawing].spines['bottom'].set_color('w')
                axs[ind_drawing].spines['top'].set_color('w') 
                axs[ind_drawing].spines['right'].set_color('w')
                axs[ind_drawing].spines['left'].set_color('w')
                axs[ind_drawing].set_title(f'Artist label: {true_category}', color='none', size=8.5, weight='bold', y=-0.155)
                for line in axs[ind_drawing].lines: line.remove()
                    
            for ind_drawing in range(np.min([max_num_misclass, len(X_draw)])):
                pred_category = categories[np.argmax(y_draw[ind_drawing])]
                axs[ind_drawing].set_title(f'Artist label: {true_category}', size=8.5, weight='bold', y=-0.155, color='dimgray')
                axs[ind_drawing].spines['bottom'].set_color('k')
                axs[ind_drawing].spines['top'].set_color('k') 
                axs[ind_drawing].spines['right'].set_color('k')
                axs[ind_drawing].spines['left'].set_color('k')
                for stroke in X_draw[ind_drawing]:
                    axs[ind_drawing].plot(stroke[0], stroke[1], color='k')
                fig.suptitle(f'DAISY says: {pred_category}',
                             color='mediumvioletred', size=16, weight='bold', y=0.875)
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
def draw_label_probs(test_dataset, y_predict, category, index=None):

    plt.close()

    if index is None: index = 0

    for_index = 0
    for count, d in enumerate(test_dataset):
        raw_drawing, data, true_label = d
        if category != '':
            if categories[true_label] == category:
                if for_index >= index: break
                for_index += 1

    palette = sn.color_palette("husl", 8)

    fig, axs = plt.subplots(1, 2, figsize=(8,4))
    for stroke in raw_drawing:
        axs[0].plot(stroke[0], stroke[1], color='k', linewidth=3)
    if category != '': axs[0].set_title(f'Artist label: {categories[true_label]}', weight='bold', y=-0.1, size=16, color='dimgray')
    axs[0].text(0.5, 1.03, f'DAISY says: {categories[np.argmax(y_predict[count])]}', weight='bold',
                size=16, color='mediumvioletred', ha='center', transform=axs[0].transAxes)
    axs[1].barh(categories, y_predict[count], color='mediumvioletred')
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[1].tick_params(left=False)
    axs[1].set_xlim(0,1)
    axs[0].invert_yaxis()
    axs[1].invert_yaxis()
    for y, x in enumerate(y_predict[count]):
        axs[1].annotate(f"   {x*100:.1f}%", xy=(x, y), va='center')
    fig.tight_layout()
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['bottom'].set_visible(False)
    axs[1].get_xaxis().set_ticks([])
    axs[1].xaxis.set_tick_params(size=15, color='mediumvioletred')
    plt.show()


def draw_label_probs_student(test_dataset, y_predict, category, index=None):

    def plotty(tog, ax, y_predict, count, text):

        if tog:

            ax.barh(np.array(categories),
                    y_predict[count], color='mediumvioletred')
            for y, x in enumerate(y_predict[count]):
                ax.annotate(f"   {x*100:.1f}%", xy=(x, y), va='center')
            text.set_color('mediumvioletred')

    plt.close()

    if index is None: index = 0

    for_index = 0
    for count, d in enumerate(test_dataset):
        raw_drawing, data, true_label = d

    fig, axs = plt.subplots(1, 2, figsize=(8,4))
    for stroke in raw_drawing:
        axs[0].plot(stroke[0], stroke[1], color='k', linewidth=3)
    text = axs[0].text(0.5, 1.03, f'DAISY says: {categories[np.argmax(y_predict[count])]}', weight='bold',
                       size=16, color='w', ha='center', transform=axs[0].transAxes)
    axs[1].barh(np.array(categories), [0.0] * len(categories), color='mediumvioletred')
    axs[1].invert_yaxis()
    tog = widgets.ToggleButton(value=False, description='DAISY says...')
    widgets.interact(plotty, tog=tog, ax=widgets.fixed(axs[1]),
                     y_predict=widgets.fixed(y_predict), count=widgets.fixed(count), text=widgets.fixed(text))

    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[1].tick_params(left=False)
    axs[1].set_xlim(0,1.2)
    axs[0].invert_yaxis()
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

    plt.close()

    for count, data in enumerate(test_dataset):
        if count == index:
            raw_data, data, label = data

    fig, ax = plt.subplots(figsize=(5,4))

    for stroke in raw_data:
        plt.plot(stroke[0], stroke[1], color='k', linewidth=3)

    true_label = categories[label]
    predicted_label = categories[np.argmax(y_predict[index])]
    plt.title(f'Artist label: {true_label}', size=16, weight='bold', y=-0.1, color='dimgray')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.invert_yaxis()

    def label_plot(var):
        if var:
            plt.text(0.5, 1.03, f'DAISY says: {predicted_label}', weight='bold', size=16, color='mediumvioletred', ha='center',
                     transform=ax.transAxes)
        fig.canvas.draw_idle()

    widgets.interact(label_plot, var=widgets.ToggleButton(value=False, description='DAISY says...'))


"""Draws a bunch of images from a certain category.

"""
def draw_grid_widget(data, category, n_rows=3, n_cols=8, start_index=None):

    plt.close()

    category_data = data.filter(lambda raw_data, data, label: label == categories.index(category))

    if start_index is None: start_index = 0

    iterator = iter(category_data)
    i = 0
    while i < start_index:
        next(iterator)
        i += 1

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))
    rectangles = []
    plt.suptitle(f'Artist label: {category}', weight='bold', y=0.1, size=16, color='dimgray')
    for i in range(n_rows):
        for j in range(n_cols):
            raw_data, simple_data, label = next(iterator)
            sketch_num = start_index + i * n_cols + j
            for line in raw_data:
                axs[i,j].plot(line[0], line[1], color='k', linewidth=2)
            axs[i,j].set_xticks([])
            axs[i,j].set_yticks([])
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

    plt.close()

    category_data = data.filter(lambda raw_data, data, label: label == categories.index(category))

    if index is None: index = 0

    for i, d in enumerate(category_data):
        if i >= index: break

    plt.figure(figsize=(5,4))
    ax = plt.gca()
    raw_data, simple_data, label = d
    for stroke in raw_data:
        plt.plot(stroke[0], stroke[1], color='k', linewidth=3)
    plt.title(f'Artist label: {categories[label]}', weight='bold', size=16, y=-0.1, color='dimgray')
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()