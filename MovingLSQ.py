import numpy as np
import functools
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares


def Error_2D(x0, src, dst, weight_map):
#
#   x0 is [x1, y1, x2, y2, x3, y3........]
#   src/dst is label point
#
    pts = x0.reshape([-1, 2])
    npoints = pts.shape[0]
    label_num = src.shape[0]
    distance = np.zeros([npoints, label_num], np.float)

    for i in range(label_num):
        label = dst[i, :]
        distance[:, i] = np.linalg.norm(pts - label, axis = 1)

    total = weight_map * distance
    error = np.sum(total, axis = 1)
    return error

class MovingLSQ_2D:
    def __init__(self, src, dst):
        #
        # Both src and dst are N x 2 numpy array, they are labeled by user
        #  
        [h, w] = src.shape
        self._label_num = h
        self._src = src
        self._dst = dst
    
    def Run(self, srcPts, alpha = 0.5):
        #
        # srcPts is [[x1, 2], [x2, y2].....] numpy array
        #
        npoints = srcPts.shape[0]
        label_num = self._label_num
        weight = np.zeros([npoints, label_num], np.float)

        for i in range(label_num):
            label = self._src[i, :]
            D = np.linalg.norm( srcPts - label , axis = 1)
            weight[:, i] = D[:]
            weight += 0.01 # prevent zero
            weight = 1 / (weight**(2*alpha))

        # weigth is npoints x label_num
        jacobian = lil_matrix((npoints, npoints*2), dtype=int)
        idx = range(npoints)
        for i in range(2):
            jacobian[idx, 2 * idx + i] = 1

        x0 = srcPts.reshape([-1])        
        result = least_squares(Error_2D, x0, jac_sparsity = jacobian, verbose = 2,
                args=(self._src, self._dst, weight))



