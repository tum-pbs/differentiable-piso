import os
import tensorflow as tf
import sys
import numpy as np
import shutil
import matplotlib.pyplot as plt
from PIL import Image
from collections.abc import Iterable


def create_base_dir(path, name):
    i = 0
    while os.path.exists(path + name + str(i).zfill(6)):
        i += 1
    try:
        os.mkdir(path + name + str(i).zfill(6))
    except:
        print("error creating directory: " + path + name + str(i))
    else:
        print("Created directory  " + path + name + str(i).zfill(6))

    return path + name + str(i).zfill(6)


def make_tf_dataset(list_tuple, mapping_func, batch_size, shuffle=True, prefetch_size=1):
    dataset = tf.data.Dataset.from_tensor_slices(list_tuple)
    if shuffle:
        dataset = dataset.shuffle(len(list_tuple[0]))
    dataset = dataset.flat_map(mapping_func)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(prefetch_size)
    return dataset


def data_path_assembler(paths, field_names, characteristics, start_frame, frame_count, step_count, dt_ratio=1):
    file_list = tuple([[] for i in range(len(field_names) + 1)])
    for p in range(len(paths)):
        pth = paths[p]
        for i in range(start_frame[p], start_frame[p] + frame_count[p] - step_count[p] * dt_ratio):
            for n in range(len(field_names)):
                file_list[n].append([pth + field_names[n] + '_' + str(i + j * dt_ratio).zfill(6) + '.npz'
                                     for j in range(0, step_count[p] + 1)])
            if isinstance(characteristics[p], Iterable):
                file_list[-1].append(characteristics[p][i - start_frame[p]])
            else:
                file_list[-1].append(characteristics[p])
    return file_list


def load_function(*data_tuple):
    output = []

    for d in range(len(data_tuple) - 1):
        output.append(np.concatenate([np.expand_dims(np.load(file)['arr_0'].astype(np.float32), axis=1)
                                      for file in data_tuple[d]], axis=1))
    output.append(np.expand_dims(np.array(data_tuple[-1]), 0).astype(np.float32))
    return tuple(output)


def load_function_wrapper(*data_tuple):
    output_tuple = tf.py_func(
        load_function, list(data_tuple), tuple([tf.float32 for i in range(len(data_tuple))]))
    output_tuple = tuple([output_tuple[i] for i in range(len(data_tuple))])
    return tf.data.Dataset.from_tensor_slices(output_tuple)


def save_source(file, path, filename):
    shutil.copy(file, path + filename)
    print('Sourcefile saved to ' + path + filename)