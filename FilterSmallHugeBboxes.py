import os
import cv2 as cv

ALL_DIR = 'AllData'
AREA_THRESHOLD = 0

for video in os.listdir(ALL_DIR):
    frames = sorted(os.listdir(os.path.join(ALL_DIR, video)))[0::2]
    labels = sorted(os.listdir(os.path.join(ALL_DIR, video)))[1::2]
    for frame, label in zip(frames, labels):
        img_h, img_w, _ = cv.imread(os.path.join(ALL_DIR, video, frame)).shape
        with open(os.path.join(ALL_DIR, video, label), 'r') as f:
            bboxes = f.read().split('\n')
            bboxes.remove('')
            bboxes = [bbox.split(' ') for bbox in bboxes]
            bboxes = [[int(c), float(x), float(y), float(w), float(h)] for c, x, y, w, h in bboxes]
            bboxes = [[c, x, y, w, h] for c, x, y, w, h in bboxes if w * h > AREA_THRESHOLD / (img_w * img_h)]
            f.close()
        with open(os.path.join(ALL_DIR, video, label), 'w') as f:
            for c, x, y, w, h in bboxes:
                f.write(f'{c} {x} {y} {w} {h}\n')
            f.close()
