#!/bin/env/python

import os
import shutil


def ls(dir='.'):
    return os.listdir(dir)


def is_hidden(path):
    return os.path.basename(path).startswith('.')


def is_visible(path):
    return not is_hidden(path)


def join_paths(dir, contents):
    return [os.path.join(dir, f) for f in contents]


def files_matching(dir, prefix=None, suffix=None, abs_paths=False,
                   only_files=False, only_dirs=False):
    files = os.listdir(dir)
    if prefix:
        files = filter(lambda f: f.startswith(prefix), files)
    if suffix:
        files = filter(lambda f: f.endswith(suffix), files)
    if only_files or only_dirs:
        paths = join_paths(dir, files)
        if only_files:
            newFiles = []
            for f, path in zip(files, paths):
                if os.path.isfile(path):
                    newFiles.append(f)
            files = newFiles
        if only_dirs:
            newFiles = []
            for f, path in zip(files, paths):
                if os.path.isdir(path):
                    newFiles.append(f)
            files = newFiles
    if abs_paths:
        files = join_paths(dir, files)
    return files


def list_subdirs(dir, startswith=None, endswith=None, abs_paths=False):
    return files_matching(dir, startswith, endswith, abs_paths,
                          only_dirs=True)


def list_files(dir, startswith=None, endswith=None, abs_paths=False):
    return files_matching(dir, startswith, endswith, abs_paths,
                          only_files=True)


def list_hidden_files(dir, startswith=None, endswith=None, abs_paths=False):
    contents = files_matching(dir, startswith, endswith, abs_paths,
                              only_files=True)
    return [f for f in contents if is_hidden(f)]


def list_visible_files(dir, startswith=None, endswith=None, abs_paths=False):
    contents = files_matching(dir, startswith, endswith, abs_paths,
                              only_files=True)
    return [f for f in contents if is_visible(f)]


def remove(path):
    if os.path.exists(path):
        try:
            os.remove(path)
        except (OSError):
            shutil.rmtree(path)


def force_create_dir(dir):
    if os.path.exists(dir):
        remove(dir)
    os.makedirs(dir)


def ensure_dir_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def basename(f, noext=False):
    name = os.path.basename(f)
    if noext:
        name = name.split('.')[0]
    return name
