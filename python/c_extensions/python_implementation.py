

import numpy as np

__author__ = 'jeromethai'


def proj_PAV(y, w=None):
    """PAV algorithm with box constraints
    """
    #y, w, l, u = s

    # if y.size != w.size:
    #     print y
    #     print w
    #     raise Exception("Shape of y (%s) != shape of w (%d)" % (y.size, w.size))

    n = len(y)
    if w is None: w = np.ones(n)
    y = y.astype(float)
    # x=y.copy()
    x=y

    if n==2:
        if y[0]>y[1]:
            x = (w.dot(y)/w.sum())*np.ones(2)
    elif n>2:
        j=range(n+1) # j contains the first index of each block
        ind = 0

        while ind < len(j)-2:
            if weighted_block_avg(y,w,j,ind+1) < weighted_block_avg(y,w,j,ind):
                j.pop(ind+1)
                while ind > 0 and weighted_block_avg(y,w,j,ind-1) > weighted_block_avg(y,w,j,ind):
                    if weighted_block_avg(y,w,j,ind) <= weighted_block_avg(y,w,j,ind-1):
                        j.pop(ind)
                        ind -= 1
            else:
                ind += 1

        for i in xrange(len(j)-1):
            x[j[i]:j[i+1]] = weighted_block_avg(y,w,j,i)*np.ones(j[i+1]-j[i])

    #return np.maximum(l,np.minimum(u,x))
    return x


# weighted average
def weighted_block_avg(y,w,j,ind):
    wB = w[j[ind]:j[ind+1]]
    return np.dot(wB,y[j[ind]:j[ind+1]])/wB.sum()


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