import os
import sys
sys.path.append('..')
import cv2
import time
from Equirec2Perspec import Equirec2Perspec as E2P

if __name__ == '__main__':
    equ = E2P.Equirectangular('../pano/panorama.jpg')

    for i in range(4):
        idx = i + 1
        name = 'image3/img-%d.png'%idx
        img = equ.GetPerspective(90, i * 90, -20, 720, 720)
        cv2.imwrite(name, img)

    name = 'gg.png'
    img = equ.GetPerspective(90, 0, -90, 720, 720)
    cv2.imwrite(name, img)
    cv2.namedWindow('GG')
    cv2.imshow('GG', img)
    cv2.waitKey(0)
