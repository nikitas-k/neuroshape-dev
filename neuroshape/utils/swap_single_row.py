"""
Helper function
"""
from numpy.random import randint

def swap_single_row(x):
    new_x = x
    idx = randint(0, len(x), size=2)
    new_idx = idx[::-1]
    new_x[new_idx] = x[idx]
    return new_x