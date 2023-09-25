import math
import numpy as np

import os
import random
from functools import partial
import jsonlines
import multiprocessing as mp
from rdp import rdp
from tqdm.auto import tqdm
from xml.dom import minidom
from svg.path import parse_path
import tensorflow as tf



with open(os.path.join(os.getcwd(),'categories.txt')) as f:
     categories = [line.rstrip('\n') for line in f]

"""Scales and shifts x, y coordinates of a drawing for preprocessing.

Scales and shifts so that the drawing is a square, and so that it extends
to edges in the x dimension.

Args:
  image_size:
    Size of image dimension, note that image is square, int.
  x:
    x coordinates, list of floats.
  y:
    y coordinate, list of floats.

Returns:

  Tuple (x_scaled, y_scaled) with

  x_scaled:
    scaled and shifted x coordinates, list of floats.
  y_scaled:
    scaled and shifted y coordinates, list of floats.
"""
def scale_and_shift(image_size, x, y):

    x_min = np.min(x)
    y_min = np.min(y)
    scaler = image_size / np.max(x)

    return (x - x_min) * scaler, (y - y_min) * scaler

"""Preprocesses raw Quick, Draw! drawing vector data.

Takes raw vector drawing data from the Quick, Draw! Google dataset (it has
been slightly preprocessed to contain only x, y coordinates of points in each
stroke), compresses into a single stroke, runs the Ramer-Douglas Puecker
algorithm on this stroke to simplify it, and zero-pads to the maximum number
of points in any drawing.

Args:
  max_n_pts:
    Maximum number of coordinate points in a single drawing, int.
  drawing:
    Raw drawing data x, y coordinate points.

Returns:

  Preprocessed data in array of size (2, max_n_pts).
"""
def drawing_process(max_n_pts, image_size, raw_data):

    drawing, label = raw_data

    vec_x = []
    vec_y = []

    for vector in drawing:

        vector[0] = [float(v) for v in vector[0]]
        vector[1] = [float(v) for v in vector[1]]
        vec_x = vec_x + vector[0]
        vec_y = vec_y + vector[1]

    rdp_vector = rdp(np.array([vec_x, vec_y]).T, algo="iter", epsilon=2)

    x, y = scale_and_shift(image_size, rdp_vector[:,0], rdp_vector[:,1])

    image_len = np.min([len(x), max_n_pts])
    binary_image = [[0.0]*image_len,[0.0]*image_len]
    binary_image[0][:image_len] = x[:image_len]
    binary_image[1][:image_len] = y[:image_len]
    binary_image = (np.array(binary_image).T).tolist()

    return {"raw_data" : drawing, "data" : binary_image, "label" : categories.index(label)}


"""Preprocesses raw Quick, Draw! drawing vector data.

Wrapper function for parallelization of drawing_process across vectors of
multiple drawings.

Args:
  max_n_pts:
    Maximum number of coordinate points in a single drawing, int.
  raw_data:
    List of raw drawing data x, y coordinate points.

Returns:

  Preprocessed data in a numpy array with each entry of size (2, max_n_pts).
"""
def vector_process(max_n_pts, image_size, raw_data, labels):

    drawing_process_wrapper = partial(drawing_process, max_n_pts, image_size)

    with mp.Pool(mp.cpu_count()) as pool:
        preprocessed_data = pool.map(drawing_process_wrapper, tqdm(list(map(list, zip(raw_data, labels)))))

    return preprocessed_data


def preprocess_student_data(max_n_pts, image_size, sketch_id, category):

    svg_dom = minidom.parse(f'./data/student/{category}/{sketch_id}.svg')

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
            x_trans.append(float(stack_trans[0][0]))
            y_trans.append(float(stack_trans[1][0]))

        x_full = x_full + x_trans
        y_full = y_full + y_trans

        X_full[count].append(x_full)
        X_full[count].append(y_full)
        count = count + 1

    X_student = vector_process(max_n_pts, image_size, [X_full], [category])

    dataset = tf.data.Dataset.from_tensors((tf.ragged.constant(X_student[0]['raw_data']),
                                            tf.constant(X_student[0]['data']),
                                            tf.constant(X_student[0]['label'])))

    return dataset


def preprocess_raw_data(max_n_pts, image_size, raw_data_dir, preprocess_data_dir, n_per_class, preprocess=True):

    if preprocess:
        total_vector_data = []
        for cat in categories:
            count = 0
            vector_data = []
            label_data = []
            print(f'[PREPROCESSING {cat} data]')
            with jsonlines.open(f'{raw_data_dir}/{cat}.ndjson') as reader:
                for obj in reader:
                    vector_data.append(obj['drawing'])
                    label_data.append(cat)
                    n_pts = 0
                    for stroke in obj['drawing']:
                        n_pts += len(np.array(stroke).T)
                    if count == (n_per_class-1): break
                    count += 1
            total_vector_data = total_vector_data + vector_process(max_n_pts, image_size, vector_data, label_data)
        with jsonlines.open(f'{preprocess_data_dir}/preprocessed_{n_per_class}-per-class.ndjson', 'w') as writer:
            writer.write_all(random.sample(total_vector_data, n_per_class * len(categories)))
