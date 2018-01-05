#!/usr/bin/env python
# -*- coding: utf-8 -*-
#================================================================
#   God Bless You.
#
#   file name: generator.py
#   author: klaus
#   email: klaus.cheng@qq.com
#   created date: 2018/01/04
#   description:
#
#================================================================

import os
import sys
import keras
import numpy as np
from keras.preprocessing.image import Iterator

from .utils.image import resize_image, load_image


class LabelFileIterator(Iterator):
    """iterate data from label file"""

    def __init__(self,
                 label_file_path,
                 image_data_generator,
                 batch_size=32,
                 num_classes=2,
                 keep_aspect_ratio=True,
                 min_side=600,
                 max_side=1024,
                 shuffle=True,
                 seed=None,
                 target_size=None,
                 preprocess_function=None):
        """TODO: to be defined1.

        Args:
            label_file_path (TODO): TODO
            image_data_generator (TODO): TODO

        Kwargs:
            batch_size (TODO): TODO
            num_classes:
            keep_aspect_ratio (TODO): TODO
            min_side (TODO): TODO
            max_side (TODO): TODO
            shuffle (TODO): TODO
            seed (TODO): TODO
            target_size (TODO): TODO
            preprocess_function:


        """

        self._label_file_path = label_file_path
        self._image_data_generator = image_data_generator
        self._batch_size = batch_size
        self._num_classes = num_classes
        self._keep_aspect_ratio = keep_aspect_ratio
        self._min_side = min_side
        self._max_side = max_side
        self._shuffle = shuffle
        self._seed = seed
        self._target_size = target_size
        self._preprocess_function = preprocess_function

        paths, labels = self._enumerate_files(self._label_file_path)
        self.paths = paths
        self.labels = labels
        self.samples = len(labels)

        super(LabelFileIterator, self).__init__(self.samples, self._batch_size,
                                                self._shuffle, self._seed)

    def _enumerate_files(self, label_file_path):
        """get file paths

        Args:
            label_file_path (TODO): TODO

        Returns: TODO

        """
        paths = []
        labels = []
        with open(label_file_path, 'r') as fld:
            for l in fld:
                x, y = l.split()
                y = int(y)
                paths.append(x)
                labels.append(y)
        return paths, labels

    def _preprocess_image(self, img, label):
        """preprocess image

        Args:
            img: numpy array image

        Returns:
            img: an numpy array image
            img_scale: the img resize factor (scale_h, scale_w). scale_h is the resize factor along the rows and scale_w along the cols

        """
        if self._preprocess_function != None:
            img, label = self._preprocess_function(img, label)
        img = self._image_data_generator.random_transform(img)
        img, img_scale = resize_image(
            img,
            min_side=self._min_side,
            max_side=self._max_side,
            target_size=self._target_size)
        return img, label, img_scale

    def _load_single_example(self, index):
        """load one example of index

        Args:
            index (TODO): TODO

        Returns: TODO

        """
        try:
            path = self.paths[index]
            label = self.labels[index]
            img = load_image(path)
            img, label, img_scale = self._preprocess_image(img, label)
        except Exception, e:
            print e
            index = np.random.randint(self.samples)
            img, label, img_scale = self._load_single_example(index)
        return img, label, img_scale

    def _get_batches_of_transformed_samples(self, index_array):
        """

        Args:
            index_array (TODO): TODO

        Returns: TODO

        """
        image_group = []
        label_group = []
        for i, j in enumerate(index_array):
            img, label, img_scale = self._load_single_example(j)
            image_group.append(img)
            label_group.append(label)

        # get the max image shape
        max_shape = tuple(
            max(image.shape[x] for image in image_group) for x in range(3))

        # construct an image batch and label batch
        image_batch = np.zeros(
            (self._batch_size, ) + max_shape, dtype=keras.backend.floatx())
        label_batch = keras.utils.to_categorical(
            label_group, num_classes=self._num_classes)

        # copy all images to the upper left part of the image batch object

        for image_index, image in enumerate(image_group):
            image_batch[image_index, :image.shape[0], :image.shape[1], :
                        image.shape[2]] = image

        return image_batch, label_batch

    def next(self):
        """for python 2.x
        Returns: The next batch

        """
        with self.lock:
            index_array = next(self.index_generator)
        return self._get_batches_of_transformed_samples(index_array)
