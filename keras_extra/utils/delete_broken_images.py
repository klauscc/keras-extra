#!/usr/bin/env python
# -*- coding: utf-8 -*-
#================================================================
#   God Bless You.
#
#   file name: delete_broken_images.py
#   author: klaus
#   email: klaus.cheng@qq.com
#   created date: 2018/01/04
#   description:
#
#================================================================

import os
import sys
import cv2
import numpy as np
from os import walk
from PIL import Image as pil_image

CURRENT_FILE_DIRECTORY = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(CURRENT_FILE_DIRECTORY, '..'))


def delete_broken_img(directory):
    """delete broken image in directory
    Returns: TODO

    """
    for (dirpath, dirnames, filenames) in walk(directory):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            try:
                img = pil_image.open(file_path)
                x = np.asarray(img, dtype=np.float32)
            except Exception, e:
                print e
                print file_path


if __name__ == "__main__":
    import sys
    delete_broken_img(sys.argv[1])
