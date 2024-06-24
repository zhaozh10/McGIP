import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from typing import List
import cv2
from tqdm import  tqdm
from sklearn.cluster import DBSCAN
import numpy as np
import glob
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from functools import partial
from typing import Tuple
from tqdm import tqdm


def gaze_cluster(gaze_seq: List[List[int]], H:int, W: int):
    # gaze_seq: The raw gaze sequence, represented as a list containing the [x_loc:int, y_loc:int] of each gaze point
    # H, W: the size of the diagnosed medical image

    # gaze_seq.shape=[length, 2]
    gaze_seq=np.array(gaze_seq)
    
    # x.shape=[length, 3]
    X=np.insert(gaze_seq, 2, 0, axis=1)
    # temporal embedding
    X[:,2]=np.arange(gaze_seq.shape[0])
    # normalization
    X=X/np.array([H,W,X.shape[0]])
    # clustering
    db = DBSCAN(eps=0.2, min_samples=int(X.shape[0]/10)+1).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_


    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    # each cluster is then converted into corresponding gaze heatmap
    groups=[]
    for i in range(n_clusters_):
        groups.append(gaze_seq[labels==i,:])
    
    return groups

