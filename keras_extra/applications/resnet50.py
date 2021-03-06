#!/usr/bin/env python
# -*- coding: utf-8 -*-
#================================================================
#   God Bless You.
#
#   file name: resnet50.py
#   author: klaus
#   email: klaus.cheng@qq.com
#   created date: 2018/01/08
#   description:
#
#================================================================

import keras
from keras.applications import resnet50

from .base_model import BaseModel


class ResNet50(BaseModel):
    """resnet50"""

    def __init__(self, *args, **kwargs):
        """TODO: to be defined1. """
        BaseModel.__init__(self, *args, **kwargs)

    def _setup_model(self):
        """setup ResNet50
        Returns: TODO

        """
        model_input = keras.layers.Input(self._input_shape)
        base_model = resnet50.ResNet50(
            input_tensor=model_input, include_top=False)
        x = base_model.output
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(
            self._num_classes, activation='softmax', name='global_cls')(x)
        model = keras.models.Model(inputs=base_model.input, outputs=x)
        return model
