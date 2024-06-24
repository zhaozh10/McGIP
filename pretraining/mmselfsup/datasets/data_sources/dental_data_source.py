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
            # gt_label, roi, record = extract_info('train', filename)
            # info = {'img_prefix': self.data_prefix}
            # img = np.asarray(Image.open(file["img"]).convert("L"), dtype="float32")
            # img=np.tile(np.expand_dims(img, axis=0), (3, 1, 1))
            # print(img.shape)
            gt=gtfile[int(filename.split('/')[-1][:-4]),0]
            # print(gt)
            info={'img_prefix':self.data_prefix,'img_info':{'filename':filename},'idx':int(i),'gt_label':gt}
            # info['img'] = file
            # info['img_info'] = {'filename': filename}
            # info['gt_label'] = np.array(gt_label, dtype=np.int64)
            # info['idx'] = int(i)
            data_infos.append(info)

        return data_infos