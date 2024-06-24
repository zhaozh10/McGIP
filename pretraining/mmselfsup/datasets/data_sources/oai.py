import matplotlib.pyplot as plt
import mmcv
from PIL import Image
import numpy as np
import json
from ..builder import DATASOURCES
from .base import BaseDataSource

@DATASOURCES.register_module()
class OAIsource(BaseDataSource):

    def load_annotations(self):
        assert isinstance(self.ann_file, str)
        preTrainList = json.load(open("../data/OAI/preTrain.json"))
        # preTrainList = json.load(open("../data/OAI/preTrain.json"))
        # namelist=jfile['train']
        data_infos=[]
        for i, filename in enumerate(preTrainList):
            """
            if you want to load image based on a txt file, img_info should be a reserved key of dict(info)
            """
            gt=eval(filename.split('/')[0])

            info = {'img_prefix': self.data_prefix}

            info['img_info'] = {'filename': filename}
            info['gt_label'] = np.array(gt, dtype=np.int64)
            info['idx'] = int(i)
            # print(info)
            data_infos.append(info)

        return data_infos