import numpy as np

__author__ = 'jeromethai'

# see reference: http://arxiv.org/pdf/1309.1541.pdf
# vectorized numpy implementation

def proj_simplex(y, start, end):
    """projects subvector of y in range(start, end)"""
    assert start>=0 and start<len(y) and end>0 and end<=len(y)
    if start >= end: return
    x = np.sort(y[start:end])[::-1]
    tmp = np.divide((np.cumsum(x)-1), np.arange(1, end-start+1))
    y[start:end] = np.maximum(y[start:end] - tmp[np.sum(x>tmp)-1], 0)
