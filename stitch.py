import os
import cv2
import glob
import numpy as np
import functools
import MovingLSQ as MLSQ
import scipy.ndimage as nd

refPT = np.array([
            [719, 0], [719, 180],

            [719, 360], [600, 360],

            [480, 360], [480, 285],

            [480, 210]
        ])

fromPT = np.array([
            [0, 0], [360, 90],

            [719, 180], [696, 410],

            [662, 640], [680, 520],

            [0, 476]
         ])

def SolveH(srcPts, dstPts):
    src = np.ones([8, 3], np.float)
    dst = np.ones([8, 3], np.float)

    src[:, :2] = fromPT[:, :]
    dst[:, :2] = refPT[:, :]
    #src = src.T
    #dst = dst.T

    #
    # H x src = dst
    #
    
    A = np.zeros([24, 8], np.float)
    B = dst.reshape([24])

    for i in range(4):
        idx = 3 * i
        A[idx, :3] = src[i, :]
        A[idx+1, 3:6] = src[i, :]
        A[idx+2, 6:] = src[i, :2]

    H = np.dot(np.linalg.pinv(A), B)
    H = np.hstack([H, 1]).reshape([3,3])
    
    result = np.dot(H, src.T).T[:, :2].astype(np.int).tolist()
    return H, result

if __name__ == '__main__':
    ref = np.zeros([720, 720, 3], np.uint8)
    origin = cv2.imread('/Users/fu-en.wang/Project/BirdViewStitch/ImageStitching/img/birdview2/img-2.png', cv2.IMREAD_COLOR)

    mask = np.zeros([720, 720], np.float32)
    cv2.fillConvexPoly(mask, refPT, color=1)
    #mask_ref = mask
    mask_ref = mask.astype(bool)
    mask = np.zeros([720, 720], np.float32)
    cv2.fillConvexPoly(mask, fromPT, color=1)
    #mask_from = mask
    mask_from = mask.astype(bool)
    
    x_map = np.tile(np.arange(720), [720, 1])
    y_map = np.tile(np.arange(720), [720, 1]).T
    
    count = x_map[mask_ref].shape[0]
    srcPts = np.zeros([count, 2])
    srcPts[:, 0] = x_map[mask_ref][:]
    srcPts[:, 1] = y_map[mask_ref][:]
    
    #solver = MLSQ.MovingLSQ_2D(refPT, fromPT)
    #dstPts = solver.Run2(srcPts, alpha = 1)
    '''
    np.save('Points/Pt2.npy', {
                        'srcPts': srcPts, 
                        'dstPts': dstPts, 
                        'srcLabel': refPT, 
                        'dstLabel': fromPT,
                        'srcMask': mask_ref,
                        'dstMask': mask_from
                        })
    exit()
    '''
    img = np.zeros([720, 720, 3], np.uint8)
    img_R = np.zeros([720, 720], np.uint8)
    img_G = np.zeros([720, 720], np.uint8)
    img_B = np.zeros([720, 720], np.uint8) 
    img_lst = sorted(glob.glob('ImageStitching/img/birdview2/*.png'))
    lst = sorted(glob.glob('Points/*.npy'))
    for idx, one in enumerate(lst):
        origin = cv2.imread(img_lst[idx])
        data = np.load(one).item()
        dstPts = data['dstPts']
        mask = data['srcMask']
        x = np.zeros([720, 720])
        y = np.zeros([720, 720])
        x[mask] = dstPts[:, 0]
        y[mask] = dstPts[:, 1]
        buf = cv2.remap(origin, x.astype(np.float32), y.astype(np.float32), cv2.INTER_CUBIC)
        buf_R = buf[:, :, 2]
        buf_G = buf[:, :, 1]
        buf_B = buf[:, :, 0]

        img_R[mask] = buf_R[mask]
        img_G[mask] = buf_G[mask]
        img_B[mask] = buf_B[mask]
    img[:, :, 0] = img_B
    img[:, :, 1] = img_G
    img[:, :, 2] = img_R

    cv2.namedWindow('GG')
    cv2.imshow('GG', img)
    cv2.waitKey(0)

