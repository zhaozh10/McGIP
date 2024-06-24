import glob
from dtw import dtw,accelerated_dtw
import numpy as np
import cv2 as cv
from functools import partial
from typing import Tuple, List

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


def hum(img: np.ndarray) -> Tuple[float, float]:
    '''Hu moments'''
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


def extract_cluster_mu_Hu(idx,namelist):
    address='../data/OAI/heatCluster/'+namelist[idx]
    namelist=sorted(glob.glob(f'{address}/*.jpg'))
    size=len(namelist)
    mu_seq=np.zeros(size)
    Hu_seq = np.zeros(size)
    for i in range(size):
        jmg = cv.resize(cv.imread(namelist[i])[:, :, 0], (32, 32))
        mu, Hu = hum(jmg)
        mu_seq[i]=mu
        Hu_seq[i]=Hu
    return mu_seq,Hu_seq


def cluster_wise_diff(x,y):
    sim=(0.7*calc_diff(x[0],y[0])+0.3*calc_diff(x[1],y[1]))
    return sim



def STGM_Similarity(idx: int,jdx: int,namelist: List[str]):

    extract = extract_cluster_mu_Hu
    mu_i, h_i = extract(idx, namelist)
    mu_j, h_j = extract(jdx, namelist)
    # reshape then concat
    mu_i = np.array(mu_i).reshape(-1, 1)
    mu_j = np.array(mu_j).reshape(-1, 1)
    h_i = np.array(h_i).reshape(-1, 1)
    h_j = np.array(h_j).reshape(-1, 1)
    x = np.concatenate((mu_i, h_i), axis=1)
    y = np.concatenate((mu_j, h_j), axis=1)
    # compute difference through DTW
    d, cost_matrix, acc_cost_matrix, path = accelerated_dtw(x, y, dist=cluster_wise_diff)
    # from diff to similarity
    sim = 1 - d / len(path[0])
    return sim


