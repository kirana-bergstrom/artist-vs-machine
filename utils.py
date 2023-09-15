import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import widgets
from PIL import Image


"""Draws a grid of images from a folder with labels.

This is a utility function that we used to draw some of the gridded images
from the notebook and tutorial. Any additional use of this function will
require the image directory to be set up in the same way as cwd/images/animals
(i.e. needs a labels.txt file, needs 12+ images labeled 000, 001, etc) and
possibly some additional editing of the function.

Args:
  directory:
    Directory name where images are located in string.
  print_matrix:
    If True prints matrix to standard output.
"""
def image_grid(directory):

    with open(f'{directory}/labels.txt', 'r') as file:
        labels = file.readlines()

    n_rows = 3
    n_cols = 4
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(2*n_cols, 2*n_rows))
    for i in range(n_rows):
        for j in range(n_cols):
            img_num = i * n_cols + j
            img = np.asarray(Image.open(f'{directory}/{img_num:03d}.jpg'))
            axs[i,j].imshow(img)
            axs[i,j].title.set_text(f'{labels[img_num]}')
            axs[i,j].axis('off')
    fig.tight_layout()
    plt.show()


def image_grid_widget(directory, num):

    with open(f'{directory}/labels.txt', 'r') as file:
        labels = file.readlines()

    n_rows = 1
    n_cols = num
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(10,3))
    for i in range(n_rows):
        for j in range(n_cols):
            img_num = i * n_cols + j
            img = np.asarray(Image.open(f'{directory}/{img_num:03d}.jpg'))
            axs[img_num].set_title(labels[img_num].replace(' ','\n').rstrip(), color='none', size=10, weight='bold', y=0.0,
                                   bbox=dict(facecolor='none', edgecolor='none'))
            axs[img_num].imshow(img)
            axs[img_num].axis('off')
    fig.tight_layout()

    def label_plot(var):
        if var:
            for i in range(n_rows):
                for j in range(n_cols):
                    img_num = i * n_cols + j
                    axs[img_num].set_title(labels[img_num].replace(' ','\n').rstrip(), color='k', size=10, weight='bold', y=0.0,
                                           bbox=dict(facecolor='w', edgecolor='black'))
        fig.canvas.draw_idle()

    widgets.interact(label_plot, var=widgets.ToggleButton(value=False, description='Label!'))
