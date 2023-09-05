# 3ClassDataset
Dataset created with 3 classes from intern annotated videos containing 11 classes.

Order of running the scripts

If want to create a subset then run:
<code>python CreateSubset.py</code>

<code>python FilterDarkImages.py</code>

<code>python FilterSmallHugeBboxes.py</code>

<code>python FilterFPS.py</code>

<code>python ChangeClasses.py</code>

<code>python BalanceClasses.py</code>

<code>python CreateYolov8Data.py</code>

<code>python TrainYolov8.py</code>

If want to use full data change <code>ALL_DIR</code> to path of AllData folder containing subfolder of video frames and labels:

<code>python FilterDarkImages.py</code>

<code>python FilterSmallHugeBboxes.py</code>

<code>python FilterFPS.py</code>

<code>python ChangeClasses.py</code>

<code>python BalanceClasses.py</code>

<code>python CreateYolov8Data.py</code>

<code>python TrainYolov8.py</code>
