#!/usr/bin/env python
# -*- coding: utf-8 -*-

import dlib
import cv2
import argparse
from PIL import Image
import numpy as np
import os, fnmatch, sys, errno
from skimage import io
from multiprocessing import Pool
import skvideo.io
from skimage import transform
from functools import partial
import time

upper_lip = [49, 50, 51, 52, 53, 54, 55, 65, 64, 63, 62, 61]
lower_lip = [49, 61, 68, 67, 66, 65, 55, 56, 57, 58, 59, 60]

lip_counters = [upper_lip + [49], lower_lip + [49]]


def generate_img_mask(img, shape):
    """ given img and predicted shape, generate the mask of lip regions.

    Args:
        img: the full image
        shape: dlib predicted 68 points.
    Returns: the same shape as input img. The mask is 3-channel with lip regions of value (255,255,255)

    """
    components = []
    for counter in lip_counters:  # upper lip and lower lip
        mask = np.zeros_like(img, np.uint8)
        for i in np.arange(len(counter) - 1):  # connect lip counters with line
            p1 = counter[i]
            p2 = counter[i + 1]
            point1 = (shape.part(p1 - 1).x, shape.part(p1 - 1).y)
            point2 = (shape.part(p2 - 1).x, shape.part(p2 - 1).y)
            cv2.line(mask, point1, point2, (20, 20, 20), 1)
        m = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), np.uint8)
        cv2.floodFill(mask, m, (0, 0), (255, 255, 255))
        components.append(mask)

    mask_image = components[0] & components[1]
    mask_image = cv2.bitwise_not(mask_image)
    return mask_image


def find_files(directory, pattern):
    filenames = []
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                filenames.append(filename)
    return filenames


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def mask(file_path, save_dir, detector, predictor):
    filepath_wo_ext = os.path.splitext(file_path)[0]
    subdir = filepath_wo_ext.split('video')[1][1:]
    # mask_dir = filepath_wo_ext.replace('video', 'mask')
    # mouth_dir = filepath_wo_ext.replace('video', 'mouth')
    # landmark_dir = filepath_wo_ext.replace('video', 'landmark')
    mask_dir = os.path.join(save_dir, 'mask', subdir)
    mouth_dir = os.path.join(save_dir, 'mouth', subdir)
    landmark_dir = os.path.join(save_dir, 'landmark', subdir)
    if os.path.exists(mask_dir) and os.path.exists(
            mouth_dir) and os.path.exists(landmark_dir):
        print('{} already processed'.format(file_path))
        return
    mask_flag = 0
    mouth_flag = 0
    landmark_flag = 0

    if os.path.exists(mask_dir):
        mask_flag = 1
    if os.path.exists(mouth_dir):
        mouth_flag = 1
    if os.path.exists(landmark_dir):
        landmark_flag = 1

    t1 = time.time()
    videogen = skvideo.io.vreader(file_path)
    frames = np.array([frame for frame in videogen])
    t2 = time.time()

    mask_imgs = []
    mouth_imgs = []
    mouth_landmark = []
    MOUTH_WIDTH = 100
    MOUTH_HEIGHT = 50
    HORIZONTAL_PAD = 0.19

    for frame in frames:
        cv_img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        dets = detector(cv_img, 1)
        if len(dets) == 0:
            continue

        shape = predictor(cv_img, dets[0])

        up_point = []
        down_point = []
        for j in upper_lip:
            up_point.append(shape.part(j - 1).x)
            up_point.append(shape.part(j - 1).y)
        for j in lower_lip:
            down_point.append(shape.part(j - 1).x)
            down_point.append(shape.part(j - 1).y)

        # mask = get_label_image((frame.shape[1], frame.shape[0]), up_point,
        # down_point, 255)
        # mask = np.asarray(mask)
        mask = generate_img_mask(cv_img, shape)

        i = -1
        mouth_points = []
        for part in shape.parts():
            i += 1
            if i < 48:  # Only take mouth region
                continue
            mouth_points.append((part.x, part.y))
        np_mouth_points = np.array(mouth_points)

        mouth_centroid = np.mean(np_mouth_points[:, -2:], axis=0)

        mouth_left = np.min(np_mouth_points[:, :-1]) * (1.0 - HORIZONTAL_PAD)
        mouth_right = np.max(np_mouth_points[:, :-1]) * (1.0 + HORIZONTAL_PAD)

        normalize_ratio = MOUTH_WIDTH / float(mouth_right - mouth_left)

        new_img_shape = (int(frame.shape[0] * normalize_ratio),
                         int(frame.shape[1] * normalize_ratio))
        frame = transform.resize(frame, new_img_shape)
        mask = transform.resize(mask, new_img_shape)

        mouth_centroid_norm = mouth_centroid * normalize_ratio

        mouth_l = int(mouth_centroid_norm[0] - MOUTH_WIDTH / 2)
        mouth_r = int(mouth_centroid_norm[0] + MOUTH_WIDTH / 2)
        mouth_t = int(mouth_centroid_norm[1] - MOUTH_HEIGHT / 2)
        mouth_b = int(mouth_centroid_norm[1] + MOUTH_HEIGHT / 2)

        mouth_imgs.append(frame[mouth_t:mouth_b, mouth_l:mouth_r])
        mask_imgs.append(mask[mouth_t:mouth_b, mouth_l:mouth_r])
        mouth_landmark.append(mouth_points)

    # save mask images
    if mask_flag != 1:
        mkdir_p(mask_dir)
        i = 0
        for mask in mask_imgs:
            io.imsave(os.path.join(mask_dir, "{0:03d}.jpg".format(i)), mask)
            i += 1
    # save cropped mouth images
    if mouth_flag != 1:
        mkdir_p(mouth_dir)
        i = 0
        for mouth in mouth_imgs:
            io.imsave(os.path.join(mouth_dir, "{0:03d}.jpg".format(i)), mouth)
            i += 1
    # save mouth landmark coordinates into txt
    if landmark_flag != 1:
        mkdir_p(landmark_dir)
        i = 0
        for landmark in mouth_landmark:
            f = open(os.path.join(landmark_dir, "{0:03d}.txt".format(i)), 'a')
            np.savetxt(f, np.array(landmark), fmt="%d")
            f.close
            i += 1

    t3 = time.time()
    print("Processing: {}. deframe: {}s, landmark: {}s".format(
        file_path, t2 - t1, t3 - t2))


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-src', help='the videos  to be processed')
    parser.add_argument('-ext', help='the video extension name')
    parser.add_argument('-dst', help='the processed destination')
    parser.add_argument(
        '-shape_dat', help='locations of dlib shape predictor model file')
    parser.add_argument( '-par_num', help= 'parallel processing number. default is 4.', type=int, default=4) 
    return parser.parse_args()


def main():
    args = arg_parse()
    SOURCE_PATH = args.src
    SOURCE_EXTS = args.ext
    SAVE_DIR = args.dst
    par_num = args.par_num

    predictor_path = args.shape_dat
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    filenames = find_files(SOURCE_PATH, SOURCE_EXTS)
    maskfun = partial(
        mask, save_dir=SAVE_DIR, detector=detector, predictor=predictor)
    p = Pool(par_num)
    p.map(maskfun, filenames)
    # for filename in filenames:
        # maskfun(filename)


if __name__ == "__main__":
    main()
