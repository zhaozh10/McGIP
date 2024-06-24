import glob
import json
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from functools import partial
from typing import Tuple
from tqdm import tqdm
import os



def gzread(filename: str, threshold: int = -1, binary: bool = False) -> np.ndarray:
    '''Read gaze file, which is a three channel 8bit image'''
    result = cv.resize(cv.imread(filename)[:, :, 0], (32, 32))
    if binary:
        return (result > 40).astype(np.uint8)
    else:
        return result

def plot_in_row(list_of_im):
    n = len(list_of_im)
    for i in range(1, n+1):
        plt.subplot(1,n,i)
        plt.imshow(list_of_im[i-1])

def overlay_heatmap(im,gz):
    gz = cv.resize(gz,im.shape[:2])
    return cv.applyColorMap(gz,cv.COLORMAP_OCEAN)//2+im//2


def moment(img: np.ndarray, i: int, j: int) -> float:
    result = 0
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            result += (x**i)*(y**j) * img[x, y]
    return result



def central_moment(img: np.ndarray, i: int, j: int) -> float:
    '''calculate the central moment M_ij for an image'''
    M_00 = moment(img, 0, 0)
    M_01 = moment(img, 0, 1)
    M_10 = moment(img, 1, 0)
    X_avg, Y_avg = (M_10/M_00), (M_01/M_00)
    result = 0
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if img[x, y]:
                result += ((x-X_avg)**i) * ((y-Y_avg)**j) * img[x, y]
    return result


def central_normalized_moment(img: np.ndarray, i: int, j: int) -> float:

    M00 = partial(moment, i=0, j=0)
    mu = central_moment
    t = 1+(i+j)/2
    return mu(img, i, j)/M00(img)**t



class Hu_moment:
    @staticmethod
    def I_1(img):
        eta = central_normalized_moment
        return eta(img, 2, 0)+eta(img, 0, 2)

    @staticmethod
    def I_2(img):
        eta = central_normalized_moment
        return (eta(img, 2, 0)-eta(img, 0, 2))**2 + 4*(eta(img, 1, 1)**2)


def gzm(img: np.ndarray) -> Tuple[float, float]:
    '''Gaze moments'''
    M = moment
    assert len(img.shape) == 2
    return M(img, 0, 0), central_normalized_moment(img, 2, 0)+central_normalized_moment(img, 0, 2)


def calc_diff(x, y):
    if np.isnan(x) or np.isnan(y):
        return 1
    if x == y:
        return 0
    else:
        if max(x,y)==0:
            return 1
        return abs(x-y)/abs(max(x,y))




def extract_Hu(idx,namelist):
    address = './heatmaps/' + namelist[idx][6:-3] + 'png'
    jmg = cv.resize(cv.imread(address)[:, :, 0], (32, 32))
    _, Hu = gzm(jmg)
    return Hu

def extract_mu_Hu(name,heatPath):
    address=os.path.join(heatPath,name.split('/')[1][:-4])
    file_address=os.path.join(address,'0.jpg')
    # address = './heatmaps/' + namelist[idx][6:-3] + 'png'
    jmg = cv.resize(cv.imread(file_address)[:, :, 0], (32, 32))
    mu, Hu = gzm(jmg)
    return mu, Hu


def moment_wise_diff(x,y):
    sim=(0.5*calc_diff(x[0],y[0])+0.5*calc_diff(x[1],y[1]))
    return sim

def moment_diff(cache_i,cache_j):
    mu_i, h_i=cache_i
    mu_j, h_j=cache_j
    sim=1 - (0.5 * calc_diff(h_i, h_j) + 0.5 * calc_diff(mu_i, mu_j))
    return sim

def moment_seq_diff(cache_i,cache_j):

    mu_seq_i, h_seq_i = cache_i
    mu_seq_j, h_seq_j = cache_j
    size = len(mu_seq_i)
    sim_seq = np.zeros(size)
    for k in range(size):
        sim_seq[k]=1 - (0.5 * calc_diff(h_seq_i[k], h_seq_j[k]) + 0.5 * calc_diff(mu_seq_i[k], mu_seq_j[k]))
    return np.mean(sim_seq)






if __name__=="__main__":
    heatmapPrefix='../data/INBreast/heatmap/'
    ann_file='../data/INBreast/namelist.txt'
    with open(ann_file, 'r') as f:
        namelist = f.read().splitlines()


    relation = np.zeros([len(namelist), len(namelist)])
    extract=extract_mu_Hu
    diff=moment_diff
    for i in tqdm(range(len(namelist))):
        mu_i, h_i = extract(namelist[i],heatmapPrefix)
        for j in tqdm(range(i, len(namelist))):
            if (i == j):
                relation[i][j] = 1
                continue
            mu_j,h_j=extract(namelist[j],heatmapPrefix)
            sim = diff((mu_i, h_i), (mu_j, h_j))

            relation[i][j] = sim
            relation[j][i]=sim
    # save as local file for efficient implementation during pretraining
    np.save('./relation_moment', relation)


