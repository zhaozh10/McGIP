import mmcv
import numpy as np

from ..builder import DATASOURCES
from .base import BaseDataSource

def extract_info(root,item_name):
    elements = item_name.split('_')
    csv_name = elements[0]

    label = -1
    roi = np.zeros([4, 1])
    if (root == 'train'):
        if (len(elements) == 6 or len(elements) == 7):
            label = int(elements[-4][-1])
            roi = np.array(elements[-2][3:].split('-'))
        elif (len(elements) == 8 or len(elements) == 9):
            label = int(elements[-6][-1])
            roi = np.array(elements[-3][3:].split('-'))
        if (len(elements) == 6 or len(elements) == 8):
            csv_name = elements[0][len(root) + 1:] + '_' + elements[1] + '.csv'
        else:
            csv_name = elements[0][len(root) + 1:] + '_' + elements[1] + '_' + elements[2] + '.csv'
    else:
        if (len(elements) == 6):
            label = int(elements[-4][-1])
            roi = np.array(elements[-3][3:].split('-'))
        else:
            label = int(elements[-2][-1])
            roi = np.array(elements[-1][3:].split('-'))
    roi = roi.astype(np.uint8)
    record={"img":item_name,"csv":csv_name}
    return label,roi,record
@DATASOURCES.register_module()
class MammoDataSource(BaseDataSource):

    def load_annotations(self):
        assert isinstance(self.ann_file, str)
        with open(self.ann_file, 'r') as f:
            namelist = f.read().splitlines()
        data_infos=[]
        for i, filename in enumerate(namelist):
            """
            if you want to load image based on a txt file, img_info should be a reserved key of dict(info)
            """
            gt_label, roi, record = extract_info('train', filename)
            info = {'img_prefix': self.data_prefix}

            info['img_info'] = {'filename': filename}
            info['gt_label'] = np.array(gt_label, dtype=np.int64)
            info['idx'] = int(i)
            data_infos.append(info)

        return data_infos