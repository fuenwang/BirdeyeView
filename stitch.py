import cv2
import numpy as np
import functools
import scipy.ndimage as nd

refPT = np.array([
            [0, 0],
            [719, 0],
            [240, 210],
            [479, 210],
            [360, 0],
            [360, 210],
            [120, 105],
            [600, 105]
        ])

fromPT = np.array([
            [0, 0],
            [719, 0],
            [87, 452],
            [632, 452],
            [360, 0],
            [360, 452],
            [43, 226],
            [675, 226]
         ])

def Reproject(coordinates):
    global H_inv
    print coordinates
    p = np.dot(H_inv, [coordinates[0], coordinates[1], 1])
    p = p[:2] / p[2]
    print p
    return (p[0], p(1))

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
    
    H, pts = SolveH(fromPT, refPT)

    for pt in refPT:
        cv2.circle(ref, (pt[0], pt[1]), 3, (0, 255, 0))

    for pt in pts:
        cv2.circle(ref, (pt[0], pt[1]), 3, (0, 0, 255))

    for pt in fromPT:
        cv2.circle(origin, (pt[0], pt[1]), 3, (0, 255, 0))
    
    #test = cv2.warpPerspective(origin, H, (720, 720), flags=(cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP))

    cv2.namedWindow('GG')
    cv2.imshow('GG', ref)
    cv2.waitKey(0)

