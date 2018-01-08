#!/usr/bin/env python
# -*- coding: utf-8 -*-
#================================================================
#   God Bless You.
#
#   file name: evaluate.py
#   author: klaus
#   email: klaus.cheng@qq.com
#   created date: 2018/01/05
#   description:
#
#================================================================

import numpy as np
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, log_loss


def evaluate(model, record_db, batch_size, verbose=0):
    """evaluate model

    Args:
        model (TODO): TODO
        record_db (TODO): TODO
        batch_size (TODO): TODO

    Returns: TODO

    """

    def _pred_batch(imgs, labels):
        img_array = np.stack(imgs)
        batch_pred_y = model.predict(img_array)
        return batch_pred_y

    batch_imgs = []
    batch_labels = []
    labels = []
    pred_y = []
    results = None
    num_samples = 0

    for string_record in record_db.get_record_iterator():
        img, label = record_db.read_example(string_record)
        batch_imgs.append(img)
        labels.append(label)
        batch_labels.append(label)
        if len(batch_imgs) == batch_size:
            if verbose and len(pred_y) % (30 * batch_size) == 0:
                print('predicting {}th img...'.format(len(pred_y)))
            batch_pred_y = _pred_batch(batch_imgs, batch_labels)
            pred_y.extend(batch_pred_y)
            batch_imgs = []
            batch_labels = []
    if len(batch_labels) != 0:
        batch_pred_y = _pred_batch(batch_imgs, batch_labels)
        pred_y.extend(batch_pred_y)

    numerical_y = np.argmax(labels, axis=1)
    numerical_pred_y = np.argmax(pred_y, axis=1)
    loss = log_loss(numerical_y, pred_y)
    acc = accuracy_score(numerical_y, numerical_pred_y)
    class_report = classification_report(numerical_y, numerical_pred_y)
    cm = confusion_matrix(numerical_y, numerical_pred_y)
    return {
        'val_loss': loss,
        'val_acc': acc,
        'class_report': class_report,
        'cm': cm
    }
