"""
A collection of functions to control whether a condition is met.
"""


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False
