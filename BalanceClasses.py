import os
import math
import cv2 as cv
import pandas as pd
import albumentations as a

ALL_DIR = 'AllData'
AUGMENT_DIR = 'augment'
ITERATIONS = 15
TRANSFORM = a.Compose([
    a.HorizontalFlip(p=0.2),
    a.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.9, 1.1), hue=(-0.1, 0.1), p=0.2),
    a.Sharpen(alpha=(0.1, 0.3), p=0.2),
    a.ShiftScaleRotate(shift_limit=(0.1, 0.3), scale_limit=(0.0, 0.3), rotate_limit=(0.0, 0.0), interpolation=cv.INTER_CUBIC, border_mode=cv.BORDER_REFLECT_101, p=0.2),
    a.Emboss(alpha=(0.1, 0.3), strength=(0.1, 0.5), p=0.2),
    a.GaussianBlur(blur_limit=(1, 5), sigma_limit=(0.1, 0.3), p=0.2),
    ], bbox_params=a.BboxParams(format='yolo'))

classes = open('classes.txt', 'r').read().split('\n')
classes.remove('')

images = []
annots = []

for video in sorted(os.listdir(ALL_DIR)):
    frames = sorted(os.listdir(os.path.join(ALL_DIR, video)))[0::2]
    images = images + [os.path.join(ALL_DIR, video, frame) for frame in frames]
    labels = sorted(os.listdir(os.path.join(ALL_DIR, video)))[1::2]
    annots = annots + [os.path.join(ALL_DIR, video, label) for label in labels]

class_counts = {c: 0 for c in classes}
df = pd.DataFrame({'image': images, 'label': annots})

def read_labels(a):
    with open(a, 'r') as f:
        bboxes = f.read().split('\n')
        bboxes.remove('')
        bboxes = [bbox.split(' ') for bbox in bboxes]
        bboxes = [[int(c), float(x), float(y), float(w), float(h)] for c, x, y, w, h in bboxes]
        f.close()
    b = []
    for c, x, y, w, h in bboxes:
        class_counts[classes[c]] = class_counts[classes[c]] + 1
        xmin = x - (w / 2.0)
        ymin = y - (h / 2.0)
        xmax = x + (w / 2.0)
        ymax = y + (h / 2.0)
        if xmin < 0.0:
            xmin = 0.0
        elif xmin > 1.0:
            xmin = 1.0
        if ymin < 0.0:
            ymin = 0.0
        elif ymin > 1.0:
            ymin = 1.0
        if xmax < 0.0:
            xmax = 0.0
        elif xmax > 1.0:
            xmax = 1.0
        if ymax < 0.0:
            ymax = 0.0
        elif ymax > 1.0:
            ymax = 1.0
        x = (xmax + xmin) / 2.0
        y = (ymax + ymin) / 2.0
        w = xmax - xmin
        h = ymax - ymin
        b.append([x, y, w, h, c])
    return b

df['bbox'] = df['label'].apply(lambda a: read_labels(a))

print('CLASS COUNT BEFORE AUGMENTATION:\n', class_counts)

min_class = min(list(class_counts.values()))
max_class = max(list(class_counts.values()))
iterations = math.floor(max_class / min_class) if ITERATIONS is None else ITERATIONS

if not os.path.exists(AUGMENT_DIR):
    os.mkdir(AUGMENT_DIR)

aug_class_count = {c: 0 for c in classes}

def augment_images_and_labels(a, transform, folder, iteration, count, max_count):
    img = cv.imread(a['image'])
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    bboxes = a['bbox']
    aug = transform(image=img, bboxes=bboxes)
    aug_img = aug['image']
    aug_bboxes = aug['bboxes']
    bboxes = []
    for x, y, w, h, c in aug_bboxes:
        if count[classes[c]] < max_count:
            bboxes.append([c, x, y, w, h])
    if len(bboxes) > 0:
        aug_img_path = os.path.join(folder, f'aug_{iteration}_' + os.path.basename(a['image']))
        cv.imwrite(aug_img_path, aug_img)
        aug_label_path = os.path.join(folder, f'aug_{iteration}_' + os.path.splitext(os.path.basename(a['image']))[0] + '.txt')
        with open(aug_label_path, 'w') as f:
            for c, x, y, w, h in bboxes:
                aug_class_count[classes[c]] = aug_class_count[classes[c]] + 1
                f.write(f'{c} {x} {y} {w} {h}\n')
            f.close()
        return aug_img_path, aug_label_path, True
    return None, None, False

all_aug_df = pd.DataFrame({'image': [], 'label': []})
for iteration in range(iterations):
    aug_df = df.apply(lambda a: augment_images_and_labels(a, TRANSFORM, AUGMENT_DIR, iteration, class_counts, max_class), axis=1, result_type='expand')
    aug_df.columns = ['image', 'label', 'keep']
    aug_df = aug_df[aug_df['keep'] == True]
    aug_df.reset_index(drop=True, inplace=True)
    aug_df = aug_df.drop(columns=['keep'])
    all_aug_df = pd.concat([all_aug_df, aug_df], axis=0, ignore_index=True)
    print(f'ITERATION {iteration} DONE')

print('ADDED COUNTS:\n', aug_class_count)

all_aug_df.to_csv('aug_df.csv', index=False)
