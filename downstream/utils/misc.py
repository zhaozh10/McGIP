import numpy as np
import torch
def retrieveCase(imgList:list)->list:
    caseList=[]
    for imgName in imgList:
        caseList.append(imgName.split('/')[-1].split('.')[0][:-1])
    return caseList

def exclusiveDataSplit(imgList: list,dataSize: int, multiCaseLists: list)-> list:
    dataSet=[]
    for imgName in imgList:
        case=imgName.split('/')[-1].split('.')[0][:-1]
        valid=True
        for caseList in multiCaseLists:
            valid=valid and (case not in caseList)
        if valid:
            dataSet.append(imgName.split('/')[-2]+'/'+imgName.split('/')[-1])
    
    return dataSet[:dataSize]

def randomSeed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False