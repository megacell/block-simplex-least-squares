import numpy as np

def block_isotonic_regression(x, ir, block_sizes, blocks_start, blocks_end):
    box = lambda y: np.maximum(np.minimum(y, 1), 0)
    inputs = [(np.arange(bs),x[s:e]) for (bs,s,e) in \
              zip(block_sizes-1,blocks_start,blocks_end)]
    proj_blocks = [ir.fit_transform(xs,ys) if xs.size > 1 else box(ys) \
                   for (xs,ys) in inputs if xs.size > 0]
    value = np.concatenate(proj_blocks)
    projected_value = box(value)
    return projected_value
