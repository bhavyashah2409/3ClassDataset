import os

ALL_DIR = 'Subset'
REDUNDANT = 'Redundant'

if not os.path.exists(REDUNDANT):
    os.mkdir(REDUNDANT)

VIDEO_INTERVAL_DICT = {
    'video 1': 12, 'video 4': 12, 'video 5': 12, 'video 6': 15, 'video 7': 12, 'video 9': 15, 'video 10': 12,
    'video 12': 15, 'video 29': 12, 'video 30': 15, 'video 31': 15, 'video 32': 12, 'video 33': 12, 'video 35': 12,
    'video 36': 12, 'video 37': 12, 'video 38': 12, 'video 39': 12, 'video 40': 15, 'video 41': 15, 'video 42': 15
    }

for video in sorted(os.listdir(ALL_DIR)):
    if video in VIDEO_INTERVAL_DICT:
        frames = sorted(os.listdir(os.path.join(ALL_DIR, video)))[0::VIDEO_INTERVAL_DICT[video]]
        labels = sorted(os.listdir(os.path.join(ALL_DIR, video)))[1::VIDEO_INTERVAL_DICT[video]]
        for frame, label in zip(sorted(os.listdir(os.path.join(ALL_DIR, video)))[0::2], sorted(os.listdir(os.path.join(ALL_DIR, video)))[1::2]):
            if frame not in frames:
                os.rename(os.path.join(ALL_DIR, video, frame), os.path.join(REDUNDANT, frame))
            if label not in labels:
                os.rename(os.path.join(ALL_DIR, video, label), os.path.join(REDUNDANT, label))
