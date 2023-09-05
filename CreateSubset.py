import os
import shutil as s

FOLDER = 'AllData'
SUBSET = 'Subset'
START = 100
END = 1000
SKIP = 10

if not os.path.exists(SUBSET):
    os.mkdir(SUBSET)

for video in sorted(os.listdir(FOLDER)):
    if not os.path.exists(os.path.join(SUBSET, video)):
        os.mkdir(os.path.join(SUBSET, video))
    frames = sorted(os.listdir(os.path.join(FOLDER, video)))[START: END: SKIP]
    labels = sorted(os.listdir(os.path.join(FOLDER, video)))[START + 1: END + 1: SKIP]
    for frame, label in zip(frames, labels):
        s.copyfile(os.path.join(FOLDER, video, frame), os.path.join(SUBSET, video, frame))
        s.copyfile(os.path.join(FOLDER, video, label), os.path.join(SUBSET, video, label))

# import os
# import cv2 as cv

# videos = [
#     'video_1', 'video_4', 'video_5', 'video_6', 'video_7', 'video_9', 'video_10', 'video_12', 'video_29',
#     'video_30', 'video_31', 'video_32', 'video_33', 'video_35', 'video_36', 'video_37', 'video_38', 'video_39',
#     'video_40', 'video_41', 'video_42'
#     ]

# for video in videos:
#     cap = cv.VideoCapture(os.path.join(r"C:\Users\aicpl\ShipsDatasets\VideoDataset\videos", video + '.mp4'))
#     print(video, cap.get(cv.CAP_PROP_FPS))
