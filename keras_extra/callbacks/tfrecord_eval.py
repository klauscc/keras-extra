#!/usr/bin/env python
# -*- coding: utf-8 -*-
#================================================================
#   God Bless You.
#
#   file name: tfrecord_eval.py
#   author: klaus
#   email: klaus.cheng@qq.com
#   created date: 2018/01/02
#   description:
#
#================================================================

import os
import sys
import keras
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, log_loss
from keras.layers import Input
from keras.models import Model
from keras.callbacks import Callback

from ..tfrecords_db import TfRecordDB
from ..core.evaluate import evaluate


class TfRecordEvalCallback(Callback):
    """evaluate callback for tf records database"""

    def __init__(self,
                 model,
                 tf_record_db,
                 checkpoint_save_path=None,
                 log_save_path=None,
                 batch_size=32):
        """
        Args:
            model: instance of @keras.model.Model. The model to be evaluate. The model's input must be a @keras.layers.Input layer.
            tf_record_db: instance of  @tfrecords_db.TfRecordDB.

        Kwargs:
            checkpoint_save_path: file to save the checkpoint.
            log_save_path: file to save the log file.
            batch_size: the evaluation batch size.


        """
        Callback.__init__(self)

        self._model = model
        self._tf_record_db = tf_record_db
        self._checkpoint_save_path = checkpoint_save_path
        self._log_save_path = log_save_path
        self._batch_size = batch_size

        self._model.compile(
            loss='categorical_crossentropy',
            optimizer='sgd',
            metrics=['accuracy'])

    def on_train_begin(self, logs={}):
        """create log save file on train begin

        Kwargs:
            logs: contains the metrics of the model such as loss,acc


        """
        self.csv_file = open(self._log_save_path, 'w')
        self.csv_file.write('epoch,loss,accuracy,val_loss,val_accurary\n')

    def on_epoch_end(self, epoch, logs={}):
        """evaluate the model and save the checkpoint and log on epoch end

        Args:
            epoch: training epoch

        Kwargs:
            logs: contains the metrics of the model such as loss,acc

        """
        result = evaluate(self._model, self._tf_record_db, self._batch_size)
        # update logs
        logs.update({'epoch': epoch})
        logs.update(result)
        self.csv_file.write(
            '{epoch:02d},{loss:.5f},{acc:.5f},{val_loss:.5f},{val_acc:.5f}\n'.
            format(**logs))
        print(
            '\nepoch: {epoch:02d},loss: {loss:.5f},acc: {acc:.5f},val_loss: {val_loss:.5f},val_acc: {val_acc:.5f}\n{class_report}\n{cm}\n'.
            format(**logs))
        #save checkpoint
        self._model.save_weights(self._checkpoint_save_path.format(**logs))
