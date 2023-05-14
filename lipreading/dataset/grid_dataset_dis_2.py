#!/usr/bin/env python
# -*- coding: utf-8 -*-
#================================================================
#   God Bless You.
#
#   file name: grid_dataset.py
#   author: klaus
#   email: klaus.cheng@qq.com
#   created date: 2018/07/15
#   description:
#
#================================================================

import os
import time
from functools import partial
import tensorflow as tf
from ..util.label_util import string2char_list
import random
import numpy as np
import math

VIDEO_LENGTH = 50
VIDEO_FRAME_SHAPE = (50, 100)
GRID_EXAMPLE_FEATURES = {
#    'video_a1': tf.VarLenFeature(tf.string),
#    'video_a2': tf.VarLenFeature(tf.string),
#    'video_b1': tf.VarLenFeature(tf.string),
    'video_grid_1': tf.VarLenFeature(tf.string),
#    'video_grid_2': tf.VarLenFeature(tf.string),
    'label_grid_1': tf.FixedLenFeature([], tf.string),
#    'label_grid_2': tf.FixedLenFeature([], tf.string),
#    'align': tf.VarLenFeature(tf.int64),
#    'mask': tf.VarLenFeature(tf.string) 
#    'class_a':tf.VarLenFeature( tf.float32),
    'class_a':tf.FixedLenFeature([29], tf.float32,),

#    'class_a':tf.FixedLenSequenceFeature([], tf.float32,allow_missing=True) 

}


def read_images(images_raw, size, channel):
    """ Read raw images to tensor.
        For example T raw image will be read to Tensor of
        shape (T, h, w, channel)

    Args:
        images_raw: 1-d `string` Tensor. Each element is an encoded jpeg image.
        size: Tuple (h, w).  The image will be resized to such size.
        channel: Int. 1 will output grayscale images, 3 outputs RGB
                 images.

    Returns: 4-D `float32` Tensor. The decoded images.
    """
    i = tf.constant(0)
    image_length = tf.shape(images_raw)[0]
    images = tf.TensorArray(dtype=tf.float32, size=image_length)

    condition = lambda i, images: tf.less(i, image_length)

    def loop_body(i, images):
        """ The loop body of reading images.
        """
        image = tf.image.resize_images(
            tf.cond(
		tf.image.is_jpeg(images_raw[i]),
		lambda: tf.image.decode_jpeg(images_raw[i], channels=channel),
		lambda: tf.image.decode_png (images_raw[i], channels=channel)),
            size=size,
            method=tf.image.ResizeMethod.BILINEAR)
        images = images.write(i, image)
        return tf.add(i, 1) , images

    i, images = tf.while_loop(
        condition,
        loop_body,
        [i, images],
        back_prop=False,
        # parallel_iterations=VIDEO_LENGTH
    )
    x = images.stack()  # T x H x W x C
    x = tf.cond(tf.shape(x)[0] < 150, lambda: x, lambda: x[::2,...]   ) 
    #tf.cond(pred, fn1, fn2, name=None) Return :either fn1() or fn2() based on the boolean predicate pred
    return x


def parse_single_example(serialized_record, use_mask,mode=tf.estimator.ModeKeys.TRAIN):
    """parse serialized_record to tensors

    Args:
        serialized_record: One tfrecord example serialized.
        use_mask: Boolean. If True, the mask will be added
                  to input as the 4-th channel.

    Returns: TODO

    """
    features = tf.parse_single_example(serialized_record,
                                       GRID_EXAMPLE_FEATURES)

    #r=random.randint(1,100)
    r=random.uniform(0.8,1.0)
    video_grid_1 = features['video_grid_1']
    video_grid_1 = tf.sparse_tensor_to_dense(video_grid_1, default_value='')
    x_grid_1 = read_images(video_grid_1, VIDEO_FRAME_SHAPE, 3)
#    if mode == tf.estimator.ModeKeys.TRAIN :
#        x_grid_1 = tf.image.random_flip_left_right(x_grid_1)
#    angle = np.random.uniform(low=-50, high=50)
#    x_grid_1 = tf.contrib.image.rotate(x_grid_1, angle * math.pi / 180)
#    x_grid_1 = tf.image.random_brightness(x_grid_1, 0.5)
#    x_grid_1 = tf.image.random_contrast(x_grid_1, 0.1, 0.5 )
#    x_grid_1 = tf.image.random_hue(x_grid_1, 0.5)    #色调
#    x_grid_1 = tf.image.random_saturation(x_grid_1, 0.3, 0.5)  #饱和度
#    x_grid_1 = tf.image.central_crop(x_grid_1,r)
#    x_grid_1 = tf.image.resize_images(x_grid_1,[50,100])

#    if r >=50:
#        x_grid_1 = tf.image.flip_left_right(x_grid_1)
    x_grid_1 /= 255.0

#    video_grid_2 = features['video_grid_2']
#    video_grid_2 = tf.sparse_tensor_to_dense(video_grid_2, default_value='')
#    x_grid_2 = read_images(video_grid_2, VIDEO_FRAME_SHAPE, 3)
#    if r>=50:
#        x_grid_2 = tf.image.flip_left_right(x_grid_2)
#    x_grid_2 /= 255.0
    #mask
#    if use_mask:
#        mask = tf.sparse_tensor_to_dense(features['mask'], default_value='')
#        m = read_images(mask, VIDEO_FRAME_SHAPE, 1)
#        m /= 255.0
#        x = tf.concat([x, m], -1)

    #parse y
    y_grid_1 = features['label_grid_1']
    y_grid_1 = tf.expand_dims(y_grid_1, 0)
#    y_grid_2 = features['label_grid_2']
#    y_grid_2 = tf.expand_dims(y_grid_2, 0)
    class_a=features['class_a']
    class_a= tf.expand_dims(class_a, 0)

    
#    print('label_y',y)
    # return x,y
    inputs = {'video_grid_1': x_grid_1, 'unpadded_length': tf.shape(x_grid_1)[0:1],}  #50,切片操作，val[0:-1]，下标0表示左起第一个元素， -1表示倒数最后一个元素，
#    inputs = {'video_grid_1': x_grid_1, 'unpadded_length': tf.shape(x_grid_1)[0:1],'video_grid_2':x_grid_2,'unpadded_length_2': tf.shape(x_grid_2)[0:1],}  #50,切片操作，val[0:-1]，下标0表示左起第一个元素， -1表示倒数最后一个元素，
    targets = {'label_grid_1': y_grid_1, 'class_a':class_a}
#    targets = {'label_grid_1': y_grid_1,'label_grid_2':y_grid_2, 'class_a':class_a}
#    print('target',targets)
    return (inputs, targets)


def grid_tfrecord_input_fn(file_name_pattern,
                           mode=tf.estimator.ModeKeys.EVAL,
                           num_epochs=1,
                           batch_size=32,
                           use_mask=False,
                           num_threads=4):
    """TODO: Docstring for grid_tfrecord_input_fn.

    Args:
        file_name_pattern: tfrecord filenames

    Kwargs:
        mode: train or others. Local shuffle will be performed if train.
        num_epochs: repeat data num_epochs times.
        batch_size: batch_size.
        use_mask: Boolean. If True, the mask will be added to input as the 4-th channel.
        num_threads: Parallel thread number.

    Returns: TODO

    """
    file_names = tf.matching_files(file_name_pattern)
    dataset = tf.data.TFRecordDataset(filenames=file_names)

#    shuffle = True  if mode == tf.estimator.ModeKeys.TRAIN else False
#    if shuffle:
#        dataset = dataset.shuffle(buffer_size=100 * batch_size + 1)

    parse_func = partial(parse_single_example, use_mask=use_mask,mode=mode) #partial函数的作用就是：将所作用的函数作为partial（）函数的第一个参数，原函数的各个参数依次作为partial（）函数的后续参数，原函数有关键字参数的一定要带上关键字，没有的话，按原有参数顺序进行补充。
    dataset = dataset.map(parse_func, num_parallel_calls=num_threads)


    dataset = dataset.repeat(num_epochs)

    #dataset = dataset.batch(batch_size)
    padded_channel = 4 if use_mask else 3
    #使用dataset中的padded_batch方法来进行，参数padded_shapes #指明每条记录中各成员要pad成的形状
    dataset = dataset.padded_batch(
        batch_size,
        padded_shapes=({
            'video_grid_1': [None, None, None, padded_channel],
            'unpadded_length': [None],
#            'video_grid_2':[None, None, None, padded_channel],
#            'unpadded_length_2': [None],

        }, {
            'label_grid_1': [None],
#            'label_grid_2': [None],
            'class_a':[None,29],
        }))
    dataset = dataset.prefetch(buffer_size=10)
    return dataset
