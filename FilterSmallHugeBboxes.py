import os
import cv2 as cv

ALL_DIR = 'Subset'
AREA_THRESHOLD = 500

for video in os.listdir(ALL_DIR):
    files = sorted(os.listdir(os.path.join(ALL_DIR, video)))
    for index, img_path in enumerate(files[::2]):
        img_h, img_w, _ = cv.imread(os.path.join(ALL_DIR, video, img_path)).shape
        with open(os.path.join(ALL_DIR, video, files[(index * 2) + 1]), 'r') as f:
            bboxes = f.read().split('\n')
            bboxes.remove('')
            bboxes = [bbox.split(' ') for bbox in bboxes]
            bboxes = [[int(c), float(x), float(y), float(w), float(h)] for c, x, y, w, h in bboxes]
            f.close()
        b = []
        for c, x, y, w, h in bboxes:
            if w * h > AREA_THRESHOLD / (img_h * img_w) and x - (w / 2.0) > 0.0 and y - (h / 2.0) > 0.0 and x + (w / 2.0) < 1.0 and y + (h / 2.0) < 1.0:
                b.append([c, x, y, w, h])
        with open(os.path.join(ALL_DIR, video, files[(index * 2) + 1]), 'w') as f:
            for c, x, y, w, h in b:
                f.write(f'{c} {x} {y} {w} {h}\n')
            f.close()
