import numpy as np

__author__ = 'jeromethai'

# see reference: http://arxiv.org/pdf/1309.1541.pdf
# vectorized numpy implementation

def proj_l1ball(y, start, end):
    """projects subvector of y in range(start, end)"""
    assert start >= 0, 'start must be >= 0'
    assert start < len(y), 'start must be < len(y)'
    assert end > 0, 'end must be > 0'
    assert end <= len(y), 'end must be <= len(y)'
    if start >= end: return
    x = np.sort(y[start:end])[::-1]
    tmp = np.divide((np.cumsum(x)-1), np.arange(1, end-start+1))
    y[start:end] = np.maximum(y[start:end] - tmp[np.sum(x>tmp)-1], 0)