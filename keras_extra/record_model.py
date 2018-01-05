#!/usr/bin/env python
# -*- coding: utf-8 -*-
#================================================================
#   God Bless You.
#
#   file name: record_model.py
#   author: klaus
#   email: klaus.cheng@qq.com
#   created date: 2018/01/05
#   description:
#
#================================================================

import keras
import tensorflow as tf

from .callbacks import TfRecordEvalCallback


class RecordModel(object):
    """model wrapper for tf record input"""

    def __init__(self, model, train_record_db, val_record_db, batch_size,
                 snapshot_save_path, log_save_path):
        """ construnct function
        Args:
            model (TODO): TODO
            train_record_db (TODO): TODO
            val_record_db (TODO): TODO
            batch_size:
            snapshot_save_path:
            log_save_path:

        """
        self._model = model
        self._train_record_db = train_record_db
        self._val_record_db = val_record_db
        self._batch_size = batch_size
        self._snapshot_save_path = snapshot_save_path
        self._log_save_path = log_save_path

        self.setup_model()

    def setup_model(self):
        """setup training model
        Returns: TODO

        """
        img_tensor, label_tensor = self._train_record_db.read_record(
            batch_size=self._batch_size)
        model_input = keras.layers.Input(tensor=img_tensor)
        train_model = model(model_input)
        train_model = keras.models.Model(
            inputs=model_input, outputs=train_model)

        #compile model
        optimizer = self._model.optimizer
        loss = self._model.loss
        metrics = self._model.metrics
        train_model.compile(
            loss=loss,
            optimizer=optimizer,
            metrics=metircs,
            target_tensors=[label_tensor])
        self._train_model = train_model

    def fit(self,
            epochs=1,
            verbose=1,
            callbacks=None,
            class_weight=None,
            initial_epoch=0,
            **kwargs):
        """TODO: Docstring for fit.

        Args:
            **kwargs (TODO): TODO

        Kwargs:
            epochs (TODO): TODO
            verbose (TODO): TODO
            callbacks (TODO): TODO
            class_weight (TODO): TODO
            initial_epoch (TODO): TODO
            steps_per_epoch (TODO): TODO

        Returns: TODO

        """
        sess = keras.backend.get_session()
        evaluate_callback = TfRecordEvalCallback(
            self._model, self._val_record_db, self._snapshot_save_path,
            self._log_save_path, self._batch_size)
        train_steps = self._train_record_db.get_steps(self._batch_size)

        # Fit the model using data from the TFRecord data tensors.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        self._train_model.fit(
            epochs=epochs,
            steps_per_epoch=train_steps,
            class_weight=class_weight,
            initial_epoch=initial_epoch,
            callbacks=[evaluate_callback],
            **kwargs)
        # Clean up the TF session.
        coord.request_stop()
        coord.join(threads)
