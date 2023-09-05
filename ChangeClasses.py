import os
import pandas as pd

ALL_DIR = 'Subset'

classes = open('classes.txt', 'r').read().split('\n')
classes.remove('')

class_counts = {c: 0 for c in classes}

for video in sorted(os.listdir(ALL_DIR)):
    labels = sorted(os.listdir(os.path.join(ALL_DIR, video)))[1::2]
    for label in labels:
        with open(os.path.join(ALL_DIR, video, label), 'r') as f:
            bboxes = f.read().split('\n')
            bboxes.remove('')
            bboxes = [bbox.split(' ') for bbox in bboxes]
            bboxes = [[int(c), float(x), float(y), float(w), float(h)] for c, x, y, w, h in bboxes]
            f.close()
        with open(os.path.join(ALL_DIR, video, label), 'w') as f:
            for c, x, y, w, h in bboxes:
                if c in [8]:
                    index = 0
                    f.write(f'{index} {x} {y} {w} {h}\n')
                    class_counts[classes[index]] = class_counts[classes[index]] + 1
                if c in [0, 1, 2, 3, 4, 6, 7, 9, 10]:
                    index = 1
                    f.write(f'{index} {x} {y} {w} {h}\n')
                    class_counts[classes[index]] = class_counts[classes[index]] + 1
                elif c in [5]:
                    index = 2
                    f.write(f'{index} {x} {y} {w} {h}\n')
                    class_counts[classes[index]] = class_counts[classes[index]] + 1
            f.close()

print(class_counts)
