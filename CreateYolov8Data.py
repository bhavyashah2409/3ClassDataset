import os
import shutil as s

ALL_DIR = 'AllData'
AUGMENT_DIR = 'augment'
FOLDER = 'data'
REDUNDANT = 'Redundant'

if not os.path.exists(FOLDER):
    os.mkdir(FOLDER)
if not os.path.exists(os.path.join(FOLDER, 'train')):
    os.mkdir(os.path.join(FOLDER, 'train'))
if not os.path.exists(os.path.join(FOLDER, 'train', 'images')):
    os.mkdir(os.path.join(FOLDER, 'train', 'images'))
if not os.path.exists(os.path.join(FOLDER, 'train', 'labels')):
    os.mkdir(os.path.join(FOLDER, 'train', 'labels'))
if not os.path.exists(os.path.join(FOLDER, 'val')):
    os.mkdir(os.path.join(FOLDER, 'val'))
if not os.path.exists(os.path.join(FOLDER, 'val', 'images')):
    os.mkdir(os.path.join(FOLDER, 'val', 'images'))
if not os.path.exists(os.path.join(FOLDER, 'val', 'labels')):
    os.mkdir(os.path.join(FOLDER, 'val', 'labels'))
if not os.path.exists(os.path.join(FOLDER, 'test')):
    os.mkdir(os.path.join(FOLDER, 'test'))
if not os.path.exists(os.path.join(FOLDER, 'test', 'images')):
    os.mkdir(os.path.join(FOLDER, 'test', 'images'))
if not os.path.exists(os.path.join(FOLDER, 'test', 'labels')):
    os.mkdir(os.path.join(FOLDER, 'test', 'labels'))

all_videos = sorted(os.listdir(ALL_DIR))
train_videos = all_videos[:33]
val_videos = all_videos[33:]

for video in all_videos:
    frames = sorted(os.listdir(os.path.join(ALL_DIR, video)))[0::2]
    labels = sorted(os.listdir(os.path.join(ALL_DIR, video)))[1::2]
    for frame, label in zip(frames, labels):
        if video in train_videos:
            os.rename(os.path.join(ALL_DIR, video, frame), os.path.join(FOLDER, 'train', 'images', frame))
            os.rename(os.path.join(ALL_DIR, video, label), os.path.join(FOLDER, 'train', 'labels', label))
        elif video in val_videos:
            os.rename(os.path.join(ALL_DIR, video, frame), os.path.join(FOLDER, 'val', 'images', frame))
            os.rename(os.path.join(ALL_DIR, video, label), os.path.join(FOLDER, 'val', 'labels', label))

for index, file in enumerate(sorted(os.listdir(REDUNDANT))):
    if index % 2 == 0:
        os.rename(os.path.join(REDUNDANT, file), os.path.join(FOLDER, 'test', 'images', file))
    elif index % 2 == 1:
        os.rename(os.path.join(REDUNDANT, file), os.path.join(FOLDER, 'test', 'labels', file))

s.rmtree(REDUNDANT)
s.rmtree(ALL_DIR)

with open('custom_data.yaml', 'w') as f:
    f.write("path: 'data'\n")
    f.write("train: 'train/images'\n")
    f.write("val: 'val/images'\n")
    f.write("test: 'test/images'\n")
    f.write("\n")
    f.write("nc: 3\n")
    f.write("\n")
    f.write("names: ['MISC', 'BOAT', 'SEAMARK']")
    f.close()

if os.path.exists(AUGMENT_DIR):
    frames = sorted(os.listdir(AUGMENT_DIR))[0::2]
    labels = sorted(os.listdir(AUGMENT_DIR))[1::2]
    TRAIN_SIZE = 0.8
    size = int(TRAIN_SIZE * len(frames))
    for frame, label in zip(frames[:size], labels[:size]):
        os.rename(os.path.join(AUGMENT_DIR, frame), os.path.join(FOLDER, 'train', 'images', frame))
        os.rename(os.path.join(AUGMENT_DIR, label), os.path.join(FOLDER, 'train', 'labels', label))
    for frame, label in zip(frames[size:], labels[size:]):
        os.rename(os.path.join(AUGMENT_DIR, frame), os.path.join(FOLDER, 'val', 'images', frame))
        os.rename(os.path.join(AUGMENT_DIR, label), os.path.join(FOLDER, 'val', 'labels', label))
    s.rmtree(AUGMENT_DIR)
