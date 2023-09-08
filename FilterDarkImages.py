import os
import cv2 as cv
import numpy as np

ALL_DIR = 'AllData'
BRIGHTNESS_THRESHOLD = 7000000

for video in os.listdir(ALL_DIR):
    frames = sorted(os.listdir(os.path.join(ALL_DIR, video)))[0::2]
    labels = sorted(os.listdir(os.path.join(ALL_DIR, video)))[1::2]
    for frame, label in zip(frames, labels):
        img_path = os.path.join(ALL_DIR, video, frame)
        label_path = os.path.join(ALL_DIR, video, label)
        bboxes = open(label_path, 'r').read()
        img = cv.imread(img_path)
        if np.sum(img) < BRIGHTNESS_THRESHOLD and len(bboxes) < 2:
            os.remove(img_path)
            os.remove(label_path)
