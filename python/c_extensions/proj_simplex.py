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


def proj_multi_simplex(y, blocks):
    assert False not in ((blocks[1:]-blocks[:-1])>0), 'block indices not increasing'
    assert blocks[0]>=0 and blocks[-1]<len(y), 'indices out of range'
    for start, end in zip(blocks[:-1], blocks[1:]): proj_simplex(y, start, end)
    proj_simplex(y, blocks[-1], len(y))