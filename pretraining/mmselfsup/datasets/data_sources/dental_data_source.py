import matplotlib.pyplot as plt
import mmcv
from PIL import Image
import numpy as np
import json
from ..builder import DATASOURCES
from .base import BaseDataSource

@DATASOURCES.register_module()
class dentalSource(BaseDataSource):
    def load_annotations(self):
        assert isinstance(self.ann_file, str)
        jfile = json.load(open("data.json"))
        gtfile=np.load('./abnormal.npy')
        namelist=jfile['train']
        data_infos=[]
        for i, file in enumerate(namelist):
            """
            if you want to load image based on a txt file, img_info should be a reserved key of dict(info)
            """
            filename=file['img']
            gt=gtfile[int(filename.split('/')[-1][:-4]),0]
            info={'img_prefix':self.data_prefix,'img_info':{'filename':filename},'idx':int(i),'gt_label':gt}
            data_infos.append(info)

        return data_infos