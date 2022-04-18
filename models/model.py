import numpy as np


class Model:
    n = 0
    m = 0
    r = 0
    p = 0

    def __init__(self, n=None, m=None, r=None, p=None):
        if n is not None:
            self.n = n
        if m is not None:
            self.m = m
        if r is not None:
            self.r = r
        if p is not None:
            self.p = p
