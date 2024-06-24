from torch.utils.data import Dataset,DataLoader
import json
from torchvision import transforms as T
from PIL import Image
import os
from typing import Callable, List, Optional, Tuple, Dict 


trans={'train':[
                    T.Resize(size=224),
                    T.RandomHorizontalFlip(),
                    # T.TrivialAugmentWide(),
                    # T.AutoAugment(),
                    # T.AugMix(),
                    T.ToTensor(),
                    T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
                ],
            'val':[
                    T.Resize(size=224),
                    T.ToTensor(),
                    T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
                ]}

class OAIset(Dataset):
    def __init__(self, dataRoot: str, annFile: List[str], transforms:List=None) -> None:
        self.annFiles=annFile
        self.dataRoot=dataRoot
        self.transforms=T.Compose(transforms)
    def getImgPath(self,index):
        path=self.annFiles[index]
        imgPath=os.path.join(self.dataRoot,self.annFiles[index])
        return imgPath

    def __getitem__(self, index)-> List:
        path=self.annFiles[index]
        gt=eval(path.split('/')[0])
        img = Image.open(os.path.join(self.dataRoot,self.annFiles[index])).convert('RGB')
        xmlPath=os.path.join(self.dataRoot,self.annFiles[index].replace("png","xml"))
        if self.transforms:
            img=self.transforms(img)
        return {'img':img,'gt':gt}

    def get(self, index)-> List:
        path=self.annFiles[index]
        gt=eval(path.split('/')[0])
        img = Image.open(os.path.join(self.dataRoot,self.annFiles[index])).convert('RGB')
        xmlPath=os.path.join(self.dataRoot,self.annFiles[index].replace("png","xml"))
        if self.transforms:
            img=self.transforms(img)
        if not os.path.isfile(xmlPath):
            return {'img':img,'gt':gt}
        else:
            with open(xmlPath, 'r', encoding='utf-8') as f:
                xmlFile = f.read()
            dictInfo = xmltodict.parse(xmlFile)['annotation']
            numBox=len(dictInfo['object'])
            bboxInfo=dictInfo['object']['bndbox']
            bbox=[eval(bboxInfo['xmin']),eval(bboxInfo['ymin']),eval(bboxInfo['xmax']),eval(bboxInfo['ymax'])]
            return {'img':img,'gt':gt,'bbox':bbox}
        

        

    def __len__(self):
        return len(self.annFiles)

