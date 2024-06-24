import numpy as np
import json
from sklearn.cluster import DBSCAN
from typing  import List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import cv2
import os

def gaussian(x, sx, y=None, sy=None):
    """Returns an array of numpy arrays (a matrix) containing values between
    1 and 0 in a 2D Gaussian distribution

    arguments
    x		-- width in pixels
    sx		-- width standard deviation

    keyword argments
    y		-- height in pixels (default = x)
    sy		-- height standard deviation (default = sx)
    """

    # square Gaussian if only x values are passed
    if y == None:
        y = x
    if sy == None:
        sy = sx
    # centers
    xo = x / 2
    yo = y / 2
    # matrix of zeros
    M = np.zeros([y, x], dtype=float)
    # gaussian matrix
    for i in range(x):
        for j in range(y):
            M[j, i] = np.exp(
                -1.0 * (((float(i) - xo) ** 2 / (2 * sx * sx)) + ((float(j) - yo) ** 2 / (2 * sy * sy))))

    return M


def gazeRead(line:list)->Union[list,bool]:
    """
    csv_gaze: Gaze info recorded in the csv file.
    return: False if corresponding gaze info is null, otherwise return gaze list.
    """
    info = line.split(';')
    imgName=info[0].split('\\')[-1]
    groundTruth = info[0].split('\\')[-2]
    csvGaze=info[2]
    csvGaze=csvGaze[1:-1]
    strGaze=csvGaze.split(', ')
    if strGaze==['']:
        return False
    assert len(strGaze)%2==0, "The length of gaze info should be even, got odd encountering"
    # gazeInfo={}
    gazeSeq=[]
    for i in range(0,len(strGaze),2):
        x=eval(strGaze[i][1:])
        y=eval(strGaze[i+1][:-1])
        gazeSeq.append((x,y))
    return {"gaze":gazeSeq,"img":imgName,"case":imgName.split('.')[0][:-1],"gt":eval(groundTruth)}

def gazeSave(saveRoot:str, line: str):
    """
    dataRoot: 
    """
    res=gazeRead(line)
    if not res:
        return 
    jsonName=saveRoot+res['img'].split('.')[0]+'.json'
    with open(jsonName, 'w') as f:
        json.dump(res, f)


def gazeHeatmap(gazepoints, dispsize, gaussianwh=200, gaussiansd=None):
    gwh = gaussianwh
    gsdwh = gwh / 6 if (gaussiansd is None) else gaussiansd
    gaus = gaussian(gwh, gsdwh)
    # matrix of zeroes
    strt = int(gwh / 2)
    heatmapsize = int(dispsize[1] + 2 * strt), int(dispsize[0] + 2 * strt)
    heatmap = np.zeros(heatmapsize, dtype=np.float64)
    # create heatmap
    for i in range(0, len(gazepoints)):
        # get x and y coordinates
        x = strt + gazepoints[i][0] - int(gwh / 2)
        y = strt + gazepoints[i][1] - int(gwh / 2)
        # correct Gaussian size if either coordinate falls outside of
        # display boundaries
        if (not 0 < x < dispsize[0]) or (not 0 < y < dispsize[1]):
            hadj = [0, gwh]
            vadj = [0, gwh]
            if 0 > x:
                hadj[0] = abs(x)
                x = 0
            elif dispsize[0] < x:
                hadj[1] = gwh - int(x - dispsize[0])
            if 0 > y:
                vadj[0] = abs(y)
                y = 0
            elif dispsize[1] < y:
                vadj[1] = gwh - int(y - dispsize[1])
            # add adjusted Gaussian to the current heatmap
            try:
                heatmap[y:y + vadj[1], x:x + hadj[1]] += gaus[vadj[0]:vadj[1], hadj[0]:hadj[1]] * gazepoints[i][2]
            except:
                # fixation was probably outside of display
                pass
        else:
            # add Gaussian to the current heatmap
            heatmap[y:y + gwh, x:x + gwh] += gaus * gazepoints[i][2]
    # resize heatmap
    heatmap = heatmap[strt:dispsize[1] + strt, strt:dispsize[0] + strt]
    return heatmap

def gazeCluster(gazeSeq: List[Tuple],canvas:Optional[int]=800):
    gazeSeq=np.array(gazeSeq)
    if len(gazeSeq)==0:
        return []
    for p in gazeSeq:
        p[0] = min(canvas,p[0])
        p[1] = min(canvas, p[1])
        # x, y = p[0], p[1]
        # x = min(canvas, x)
        # y = min(canvas, y)
    gazeSeq=np.concatenate((gazeSeq,np.expand_dims(np.arange(gazeSeq.shape[0]),axis=1)),axis=1)
    X=gazeSeq
    # add time dimension
    # X=np.concatenate((gazeSeq,np.expand_dims(np.arange(gazeSeq.shape[0]),axis=1)),axis=1)
    X=X/np.array([canvas,canvas,X.shape[0]])
    
    db = DBSCAN(eps=0.2, min_samples=int(X.shape[0]/20)+1).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_


    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    partions=[]
    gazeSeq[:,2]=1
    for i in range(n_clusters_):
        partions.append(gazeSeq[labels==i,:])
        # print(i)
    # plt.figure(figsize=(9,9))
    # ax = plt.axes(projection='3d')
    # ax.scatter3D(X[:,0], X[:,2], X[:,1], color="red")
    # # ax.set_zticks(np.arange(0, 1, 0.05))
    # # plt.title("simple 3D scatter plot")
    #
    # # show plot
    # plt.show()
    return partions

def createSaveHeat(gazeSeq: List[Tuple],savePath:str, canvas:Tuple=(800,800)):
    display_height= canvas[1]
    display_width = canvas[0]
    gauss_size = int(display_height / 10)
    # hm_array为生产热度图的numpy格式
    # partitions=np.array_split(gaze_list,piece)
    # partitions=gazeCluster(gazeSeq['gaze'])
    cur_gaze=gazeSeq['gaze']
    # if len(cur_gaze)==0:
    #     shown_img = np.zeros_like(canvas)
    #     shown_img = cv2.normalize(shown_img, shown_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_16U)
    #     cv2.imwrite(os.path.join(savePath,'0.jpg'), shown_img)
    #     return
    gazeCanvas = np.zeros(canvas, dtype=np.float)
    
    # gaze = gazeRead(info[2])
    # with open(jsonName, 'w') as f:
    #     json.dump(gaze, f)
    for p in cur_gaze:
        x, y = p[0], p[1]
        x = min(799, x)
        y = min(799, y)
        gazeCanvas[y, x] = 1.0
    gazeCanvas = cv2.GaussianBlur(gazeCanvas, (199, 199), 0)
    # gazeCanvas = (gazeCanvas - np.min(gazeCanvas))
    gazeCanvas /= np.max(gazeCanvas)
    # cv2.imwrite(os.path.join(savePath,'0.jpg'), gazeCanvas)

    # for p in cur_gaze:
    #     p[0] = min(canvas[0],p[0])
    #     p[1] = min(canvas[1], p[1])
    #     # x, y = p[0], p[1]
    # gazeCanvas = cv2.GaussianBlur(gazeCanvas, (199, 199), 0)
    # gazeCanvas = (gazeCanvas - np.min(gazeCanvas))
    # gazeCanvas /= np.max(gazeCanvas)
    # cur_gaze=np.array(cur_gaze)
    # cur_gaze=np.concatenate((cur_gaze,np.expand_dims(np.ones(cur_gaze.shape[0]),axis=1)),axis=1)


    if not os.path.exists(savePath):
        os.makedirs(savePath)
    # hm_array = gazeHeatmap(cur_gaze, (display_width, display_height), gaussianwh=gauss_size,gaussiansd=gauss_size / 6)
    # hm_array = hm_array / hm_array.max()
    # shown_img = hm_array
    # shown_img = cv2.normalize(shown_img, shown_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_16U)
    # cv2.imwrite(os.path.join(savePath,'0.jpg'), shown_img)
    cv2.imwrite(os.path.join(savePath,'0.jpg'), (gazeCanvas*255).astype('int32'))
    

def createSaveCluster(gazeSeq: List[Tuple],savePath:str, canvas:Tuple=(1920,1080)):
    # piece=split_parts
    display_height= canvas[1]
    display_width = canvas[0]
    gauss_size = int(display_height / 10)
    # hm_array为生产热度图的numpy格式
    # partitions=np.array_split(gaze_list,piece)
    partitions=gazeCluster(gazeSeq['gaze'])
    # target_dir=f'./heatCluster/{savePath}/'
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    for i in range(len(partitions)):
        cur_gaze=partitions[i]
        hm_array = gazeHeatmap(cur_gaze, (display_width, display_height), gaussianwh=gauss_size,gaussiansd=gauss_size / 6)
        hm_array = hm_array / hm_array.max()
        shown_img = hm_array
        shown_img = cv2.normalize(shown_img, shown_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_16U)
        cv2.imwrite(os.path.join(savePath,f'{i}.jpg'), shown_img)
        # cv2.imwrite(f'{savePath}/{i}.png', shown_img)

        # shown_img = cv2.applyColorMap(shown_img, cmapy.cmap('twilight_shifted'))
        # plt.figure()
        # plt.imshow(shown_img[:, :, (2, 1, 0)], cmap='twilight_shifted')  # 叠加注意力热度图
        # plt.axis('off')  # 去掉坐标轴
        # plt.title(f"label {label}")

    # plt.show()