#!/usr/bin/env python
# -*- coding: utf-8 -*-
#================================================================
#   God Bless You.
#
#   file name: model.py
#   author: klaus
#   email: klaus.cheng@qq.com
#   created date: 2018/01/08
#   description:
#
#================================================================

import keras
import numpy as np
import tensorflow as tf
from keras_extra.tfrecords_db import TfRecordDB
from keras_extra.callbacks.tfrecord_eval import TfRecordEvalCallback
import keras_extra.core.evaluate as ke_eval


class BaseModel(object):
    """base model for training,evaluation on tf record"""

    def __init__(
            self,
            num_classes,
            input_shape,
            batch_size=32,
            preprocess_input=None,
            preprocess_output=None,
            weights=None,
    ):
        """TODO: to be defined1.

        Args:
            train_record_path (TODO): TODO
            val_record_path (TODO): TODO
            num_classes (TODO): TODO
            input_shape (TODO): TODO

        Kwargs:
            preprocess_func (TODO): TODO
            weights (TODO): TODO
            snapshot_save_path:
            log_save_path:


        """
        self._num_classes = num_classes
        self._input_shape = input_shape
        self._batch_size = batch_size
        self._preprocess_input = preprocess_input
        self._preprocess_output = preprocess_output
        self._weights = weights

        self._model = self._setup_model()
        if self._weights:
            self._model.load_weights(self._weights)

    def _setup_model(self):
        """set up model. Must be implemented by the sub class
        Returns keras Model

        """
        raise NotImplementedError("setup_model method not implemented")

    def _preprocess_input_and_output(self, img, label):
        """preprocess the input and output. the input and output can be numpy array or tensorflow tensors, one example or batch examples.

        Args:
            img (TODO): TODO
            label (TODO): TODO

        Returns: TODO

        """
        if self._preprocess_input:
            img = self._preprocess_input(img)
        if self._preprocess_output:
            label = self._preprocess_output(label)
        return img, label

    def _prepare_db(self):
        """prepare the train record db and val record db
        Returns: TODO

        """

        train_db = TfRecordDB(None, 'train', self._train_record_path,
                              self._preprocess_input_and_output,
                              self._num_classes, self._input_shape[0:2],
                              self._input_shape[2])
        val_db = TfRecordDB(None, 'val', self._val_record_path,
                            self._preprocess_input_and_output,
                            self._num_classes, self._input_shape[0:2],
                            self._input_shape[2])
        input_tensor, output_tensor = train_db.read_record(
            batch_size=self._batch_size)

        self._train_db = train_db
        self._val_db = val_db
        self._input_tensor = input_tensor
        self._output_tensor = output_tensor

    def _wrap_model(self):
        """wrap the model for tensorflow record
        Returns: TODO

        """
        # build train model
        model_input = keras.layers.Input(tensor=self._input_tensor)
        train_model = self._model(model_input)
        train_model = keras.models.Model(
            inputs=model_input, outputs=train_model)

        self._train_model = train_model

    def summary(self):
        """print the network summary
        Returns: TODO

        """
        return self._model.summary()

    def compile(self, snapshot_save_path=None, log_save_path=None, **kwargs):
        """compile the model

        Args:
            *args (TODO): TODO
            **kwargs (TODO): TODO

        Returns: TODO

        """
        self._model.compile(**kwargs)
        self._train_model.compile(target_tensors=self._output_tensor, **kwargs)

        self._snapshot_save_path = snapshot_save_path
        self._log_save_path = log_save_path

    def fit(self,
            train_record_path,
            val_record_path,
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
        self._train_record_path = train_record_path
        self._val_record_path = val_record_path

        self._prepare_db()
        self._wrap_model()

        sess = keras.backend.get_session()
        evaluate_callback = TfRecordEvalCallback(
            self._model, self._val_db, self._snapshot_save_path,
            self._log_save_path, self._batch_size)
        train_steps = self._train_db.get_steps(self._batch_size)

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

    def evaluate(self, record_path, prefix, weights=None):
        """evaluate

        Args:
            record_path (TODO): TODO

        Kwargs:
            weights (TODO): TODO

        Returns: TODO

        """
        eval_db = TfRecordDB(
            None, prefix, record_path, self._preprocess_input_and_output,
            self._num_classes, self._input_shape[0:2], self._input_shape[2])
        if weights != None:
            self._model.load_weights(weights)
        result = ke_eval.evaluate(
            self._model, eval_db, self._batch_size, verbose=1)
        print(
            'test_loss: {val_loss:.5f},test_acc: {val_acc:.5f}\nclassification report:\n{class_report}\nconfusion maxtrix:\n{cm}\n'.
            format(**result))

    def predict(self, path):
        """predict one image

        Args:
            path (TODO): TODO

        Returns: TODO

        """
        img = TfRecordDB._load_image(path, self._input_shape[:2])
        img = np.asarray(img, dtype=np.float32)
        img = self._preprocess_input(img)
        img_batch = np.expand_dims(img, axis=0)
        pred = self._model.predict(img_batch)
        return pred[0]
