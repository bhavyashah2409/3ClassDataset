import os
import cv2 as cv
import pandas as pd

ALL_DIR = 'AllData'
CLASS_THRESHOLD = 100

classes = open('classes.txt', 'r').read().split('\n')
classes.remove('')

class_counts = {c: 0 for c in classes}

for video in sorted(os.listdir(ALL_DIR)):
    frames = sorted(os.listdir(os.path.join(ALL_DIR, video)))[0::2]
    labels = sorted(os.listdir(os.path.join(ALL_DIR, video)))[1::2]
    for frame, label in zip(frames, labels):
        img_h, img_w, _ = cv.imread(os.path.join(ALL_DIR, video, frame)).shape
        with open(os.path.join(ALL_DIR, video, label), 'r') as f:
            bboxes = f.read().split('\n')
            bboxes.remove('')
            bboxes = [bbox.split(' ') for bbox in bboxes]
            bboxes = [[int(c), float(x), float(y), float(w), float(h)] for c, x, y, w, h in bboxes]
            f.close()
        b = []
        for c, x, y, w, h in bboxes:
            if c in [8] and w * h < CLASS_THRESHOLD / (img_w * img_h):
                b.append([0, x, y, w, h])
                class_counts['MISC'] = class_counts['MISC'] + 1
            elif c in [8] and w * h >= CLASS_THRESHOLD / (img_w * img_h):
                b.append([1, x, y, w, h])
                class_counts['BOAT'] = class_counts['BOAT'] + 1
            elif c in [0, 1, 2, 3, 4, 6, 7, 9, 10]:
                b.append([1, x, y, w, h])
                class_counts['BOAT'] = class_counts['BOAT'] + 1
            elif c in [5]:
                b.append([2, x, y, w, h])
                class_counts['SEAMARK'] = class_counts['SEAMARK'] + 1
        with open(os.path.join(ALL_DIR, video, label), 'w') as f:
            for c, x, y, w, h in b:
                f.write(f'{c} {x} {y} {w} {h}\n')
            f.close()

print(class_counts)
