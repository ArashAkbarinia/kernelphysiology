"""
A collection of functions to control whether a condition is met.
"""

import re


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def isint(value):
    try:
        int(value)
        return True
    except ValueError:
        return False


def atof(value):
    try:
        return float(value)
    except ValueError:
        return value


def atoi(value):
    try:
        return int(value)
    except ValueError:
        return value


def natural_keys(text, delimiter=None, remove=None):
    """
    alist.sort(key=natural_keys) sorts in human order
    adapted from http://nedbatchelder.com/blog/200712/human_sorting.html
    """
    if remove is not None:
        text = text.replace(remove, '')
    if delimiter is None:
        return [atoi(c) for c in re.split(r'(\d+)', text)]
    else:
        return [atof(c) for c in text.split(delimiter)]
