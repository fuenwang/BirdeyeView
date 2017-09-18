import os
import sys
sys.path.append('..')
import cv2
import glob
import numpy as np
import functools
import MovingLSQ as MLSQ
import scipy.ndimage as nd
'''
refPT = np.array([
            [0, 0], [90, 0], [180, 0], [270, 0], [360, 0], [450, 0], [540, 0], [630, 0],
            [719, 0], [659, 53], [599, 105], [539, 158],
            [479, 210], [420, 210], [360, 210], [300, 210],
            [240, 210], [210, 184], [180, 158], [120, 105], [60, 53], [30, 27]
        ])
fromPT = np.array([
            [0, 0], [90, 0], [180, 0], [270, 0], [360, 0], [450, 0], [540, 0], [630, 0],
            [719, 0], [719, 180], [719, 360], [719, 540],
            [719, 719], [540, 719], [360, 719], [180, 719],
            [0, 719], [0, 630], [0, 540], [0, 360], [0, 180], [0, 90]
        ])
'''
refPT = np.array([
            [719, 0], [719, 90], [719, 180], [719, 270], [719, 360], [719, 450], [719, 540], [719, 630],
            [719, 719], [660, 667], [600, 614], [540, 562],
            [479, 509], [479, 434], [479, 359], [479, 284],
            [479, 209], [510, 183], [540, 157], [600, 105], [660, 53], [690, 27]
        ])
fromPT = np.array([
            [0, 0], [90, 0], [180, 0], [270, 0], [360, 0], [450, 0], [540, 0], [630, 0],
            [719, 0], [719, 180], [719, 360], [719, 540],
            [719, 719], [540, 719], [360, 719], [180, 719],
            [0, 719], [0, 630], [0, 540], [0, 360], [0, 180], [0, 90]
        ])

if __name__ == '__main__':
    #refPT[:, 0] = 360 - (refPT[:, 0] - 360)
    #fromPT[:, 0] = 360 - (fromPT[:, 0] - 360)
    tmp = np.load('Points/Pt1.npy').item()
    refPT = tmp['srcLabel']
    fromPT = tmp['dstLabel']
    ref = np.zeros([720, 720, 3], np.uint8)
    origin = cv2.imread('image/img-1.png', cv2.IMREAD_COLOR)

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
    '''
    solver = MLSQ.MovingLSQ(refPT, fromPT)
    dstPts = solver.Run_Affine(srcPts)
    #dstPts = solver.Run_Rigid(srcPts)
    np.save('Points/Pt1.npy', {
                        'srcPts': srcPts, 
                        'dstPts': dstPts, 
                        'srcLabel': refPT, 
                        'dstLabel': fromPT,
                        'srcMask': mask_ref,
                        'dstMask': mask_from
                        })
    #exit()
    '''
    img = np.zeros([720, 720, 3], np.uint8)
    img_R = np.zeros([720, 720], np.uint8)
    img_G = np.zeros([720, 720], np.uint8)
    img_B = np.zeros([720, 720], np.uint8) 
    img_lst = sorted(glob.glob('image/*.png'))
    lst = sorted(glob.glob('Points/*.npy'))
    for i, one in enumerate(lst):
        idx = int(one.split('/')[-1][2]) - 1
        origin = cv2.imread(img_lst[idx])
        data = np.load(one).item()
        dstPts = data['dstPts']
        mask = data['srcMask']
        mask2 = data['dstMask']
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
    cv2.line(img, (0, 0), (243, 213), (0, 0, 0), 4)
    cv2.line(img, (719, 0), (477, 213), (0, 0, 0), 4)
    cv2.line(img, (719, 719), (477, 507), (0, 0, 0), 4)
    cv2.line(img, (0, 719), (243, 507), (0, 0, 0), 4)
    cv2.namedWindow('GG')
    cv2.imshow('GG', img)
    #cv2.imshow('GG', mask2.astype(np.float32))
    cv2.waitKey(0)
    cv2.imwrite('../gg.png', img)

