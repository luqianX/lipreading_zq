#!/usr/bin/env python
# -*- coding: utf-8 -*-
#================================================================
#   God Bless You.
#
#   file name: convert_tfrecord.py
#   author: klaus
#   email: klaus.cheng@qq.com
#   created date: 2018/11/12
#   description:
#
#================================================================

import os
import glob
import time
import argparse
import numpy as np
import tensorflow as tf

from dataseter.GRID.tf_record import feature
from lipreading.dataset.lrs_dataset import lrs_tfrecord_input_fn

EXAMPLE_FEATURES = {
    'video': tf.VarLenFeature(tf.string),
    'label': tf.FixedLenFeature([], tf.string),
}


def _convert2example(video, label):
    v_feature = {
        'video': feature.bytes_list_feature(
            tf.compat.as_bytes(video.tostring())),
        'label': feature.bytes_feature(tf.compat.as_bytes(label))
    }
    return tf.train.Example(features=tf.train.Features(feature=v_feature))


def convert_record(src_path, dst_path, img_size):
    """TODO: Docstring for convert_record.

    Args:
        src_path (TODO): TODO
        dst_path (TODO): TODO
        img_size (TODO): TODO

    Returns: TODO

    """
    bs = 10
    dataset = lrs_tfrecord_input_fn(
        file_name_pattern=src_path,
        num_epochs=1,
        mode=tf.estimator.ModeKeys.EVAL,
        batch_size=bs,
        div_255=False,
        num_threads=4)
    iterator = dataset.make_one_shot_iterator()
    feature_itr, label_itr = iterator.get_next()
    with tf.device('/cpu:0'):
        with tf.python_io.TFRecordWriter(dst_path) as writer:
            with tf.Session() as sess:
                t = 0
                while True:
                    idx = 0
                    t1 = time.time()
                    feature, label = sess.run([feature_itr, label_itr])
                    video = feature['video']
                    video = video.astype(np.uint8)
                    video_len = feature['unpadded_length']
                    for i in range(bs):
                        l = int(video_len[i])
                        unpadded_video = video[:l, ...]
                        video_label = label['label'][i][0]
                        example = _convert2example(unpadded_video, video_label)
                        writer.write(example.SerializeToString())
                    t2 = time.time()
                    t = t * idx / (idx + 1) + (t2 - t1) / (idx + 1)
                    idx += 1
                    print(' writing video {}. {:.2f}ms/ {} video'.format(
                        idx, t * 1000, bs))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-src', help='src dir')
    parser.add_argument('-dst', help='dst file save dir')
    parser.add_argument('-size', help='img_size', type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    src_files = glob.glob(args.src + "/*.tfrecord")
    img_size = [args.size, args.size]
    os.system('mkdir -p {} '.format(args.dst))

    for f in src_files:
        filename = os.path.split(f)[-1]
        dst_file = os.path.join(args.dst, filename)
        convert_record(f, dst_file, img_size)


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    main()
