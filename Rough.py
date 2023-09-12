import os
import cv2 as cv
import random as rn

ALL_DIR = 'AllData'
CLASSES = [8, 2, 3]
IMAGES = 5

videos = sorted(os.listdir(ALL_DIR))

i = 0
for video in videos:
    frames = sorted(os.listdir(os.path.join(ALL_DIR, video)))[0::2]
    labels = sorted(os.listdir(os.path.join(ALL_DIR, video)))[1::2]
    for frame, label in zip(frames, labels):
        if i == IMAGES:
            break
        with open(os.path.join(ALL_DIR, video, label), 'r') as f:
            bboxes = f.read().split('\n')
            bboxes.remove('')
            bboxes = [bbox.split(' ') for bbox in bboxes]
            bboxes = [[int(c), float(x), float(y), float(w), float(h)] for c, x, y, w, h in bboxes]
            f.close()
        for c, x, y, w, h in bboxes:
            if c in CLASSES:
                i = i + 1
                img = cv.imread(os.path.join(ALL_DIR, video, frame))
                # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                img_h, img_w, _ = img.shape
                xmin = int((x - (w / 2.0)) * img_w)
                ymin = int((y - (h / 2.0)) * img_h)
                xmax = int((x + (w / 2.0)) * img_w)
                ymax = int((y + (h / 2.0)) * img_h)
                area = w * img_w * h * img_h
                print(f'AREA: {area}')
                img = cv.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                img = cv.putText(img, f'{c}', (xmin, ymin - 10), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
                cv.imshow('Image', img)
                if cv.waitKey(0) == 27:
                    cv.destroyAllWindows()
                    break
