import cv2
import numpy as np
import functools
import MovingLSQ as MLSQ
import scipy.ndimage as nd

refPT = np.array([
            [0, 0],
            [180, 0],
            [360, 0],
            [540, 0],
            [719, 0],
            [600, 105],
            [479, 210],
            [360, 210],
            [240, 210],
            [120, 105]
        ])

fromPT = np.array([
            [0, 0],
            [180, 0],
            [360, 0],
            [540, 0],
            [719, 0],
            [675, 226],
            [632, 452],
            [360, 452],
            [87, 452],
            [43, 226]
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
    origin = cv2.imread('/Users/fu-en.wang/Project/BirdViewStitch/ImageStitching/img/birdview2/img-1.png', cv2.IMREAD_COLOR)

    mask = np.zeros([720, 720], np.float32)
    cv2.fillConvexPoly(mask, refPT, color=1)
    #mask_ref = mask
    mask_ref = mask.astype(bool)
    mask = np.zeros([720, 720], np.float32)
    cv2.fillConvexPoly(mask, fromPT, color=1)
    #mask_from = mask
    mask_from = mask.astype(bool)
    
    x_map_ref = np.tile(np.arange(720), [720, 1])
    y_map_ref = np.tile(np.arange(720), [720, 1]).T
    x_map_from = np.tile(np.arange(720), [720, 1])
    y_map_from = np.tile(np.arange(720), [720, 1]).T
    
    count = x_map_from[mask_ref].shape[0]
    srcPts = np.zeros([count, 2])
    srcPts[:, 0] = x_map_ref[mask_ref][:]
    srcPts[:, 1] = y_map_ref[mask_ref][:]
    #print srcPts
    
    #srcPts = np.array([[240, 210], [120, 105]])
    solver = MLSQ.MovingLSQ_2D(refPT, fromPT)
    #dstPts = solver.Run2(srcPts, alpha = 1)
    tmp = np.load('Pt1.npy').item()
    srcPts = tmp['srcPts']
    dstPts = tmp['dstPts']
    refPT = tmp['srcLabel']
    fromPT = tmp['dstLabel']
        #exit()
    #print (dstPts - srcPts)
    #print dstPts[-1, :]
    #exit()
    #dstPts = np.load('dstPts.npy')
    mask[:, :] = 0
    x_mask = mask.copy().astype(np.float)
    y_mask = mask.copy().astype(np.float)
    x_mask[mask_ref] = dstPts[:, 0]
    y_mask[mask_ref] = dstPts[:, 1]
    np.save('Pt1.npy', {
                        'srcPts': srcPts, 
                        'dstPts': dstPts, 
                        'srcLabel': refPT, 
                        'dstLabel': fromPT,
                        'srcMask': mask_ref,
                        'dstMask': mask_from
                        })

    #print x_mask
    img = cv2.remap(origin, x_mask.astype(np.float32), y_mask.astype(np.float32), cv2.INTER_CUBIC)
    #x_mask[]
    #dstPts = solver.Run(srcPts)
    #np.save('dstPts.npy', dstPts)
    #exit()
    for pt in refPT:
        cv2.circle(ref, (pt[0], pt[1]), 3, (0, 255, 0))


    for pt in range(count):
        cv2.circle(origin, (int(dstPts[pt, 0]), int(dstPts[pt, 1])), 2, (0, 255, 0))
    
    #test = cv2.warpPerspective(origin, H, (720, 720), flags=(cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP))

    cv2.namedWindow('GG')
    cv2.imshow('GG', img)
    cv2.waitKey(0)

