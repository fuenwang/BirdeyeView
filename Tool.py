import os
import sys
import cv2
import numpy as np

class Label:
    def __init__(self, img):
        self._img = cv2.imread(img, cv2.IMREAD_COLOR)
        self.Pts = []
    def _Click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(self._img, (x, y), 1, (0, 255, 0), 3)
            self.Pts.append([x, y])
            print [x, y]

    def Run(self):
        cv2.namedWindow('window1')
        cv2.setMouseCallback('window1', self._Click)
        show = True
        while show:
            cv2.imshow('window1', self._img)
            key = cv2.waitKey(33)
            if key == ord('q'):
                cv2.destroyAllWindows()
                show = False


if __name__ == '__main__':
    img = 'ImageStitching/img/birdview2/img-2.png'

    label = Label(img)
    label.Run()
