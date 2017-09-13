import os
import cv2
import time
from Equirec2Perspec import Equirec2Perspec as E2P

if __name__ == '__main__':
    equ = E2P.Equirectangular('../pano/pano_dqdfXHzr6Pt7z76KDohVmg.jpg')
    for i in range(7):
        if i != 0:
            img = equ.GetPerspective(60, (i-1)*60, -20, 720, 720)
            name = '/Users/fu-en.wang/Project/BirdViewStitch/ImageStitching/img/birdview2/img-%d.png'%i
            cv2.imwrite(name, img)
        else:
            pass
            #img = equ.GetPerspective(90, 0, -90, 720, 720)

    i = 7
    img = equ.GetPerspective(60, 0, -90, 720, 720)
    name = '/Users/fu-en.wang/Project/BirdViewStitch/ImageStitching/img/birdview2/img-%d.png'%i
    cv2.imwrite(name, img)
