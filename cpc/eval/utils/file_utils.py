# Copyright (c) Cochlear.ai, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
""".

Created on Fri Mar 27 16:15:31 2020
@author: skchang@cochlear.ai
"""
import os


def get_fps_from_txt(txt_fname):
    """Get filepaths from text file."""
    fps = []
    with open(txt_fname) as fp:
        fps = fp.read().splitlines()
    return fps


def check_files_exist_in_dir(file_list, dir_path):
    """Chekc files exists from a file list and a target directory."""
    for i, fname in enumerate(file_list):
        fullpath = dir_path + '/' + fname
        if os.path.exists(fullpath) is False:
            raise FileNotFoundError(fname)
    print('check_files_exist: Total {} files, ok'.format(i))
    return i


def check_files_exist(file_list):
    """Chekc files exists from a file list."""
    for i, fname in enumerate(file_list):
        fullpath = fname
        if os.path.exists(fullpath) is False:
            raise FileNotFoundError(fname)
    print('check_files_exist: Total {} files, ok'.format(i))
    return i
