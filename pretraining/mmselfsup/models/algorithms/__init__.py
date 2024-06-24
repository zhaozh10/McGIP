# Copyright (c) OpenMMLab. All rights reserved.
from .barlowtwins import BarlowTwins
from .base import BaseModel
from .byol import BYOL
from .cae import CAE
from .classification import Classification
from .deepcluster import DeepCluster
from .densecl import DenseCL
from .mae import MAE
from .mmcls_classifier_wrapper import MMClsImageClassifierWrapper
from .moco import MoCo
from .mocov3 import MoCoV3
from .npid import NPID
from .odc import ODC
from .relative_loc import RelativeLoc
from .rotation_pred import RotationPred
from .simclr import SimCLR
from .simmim import SimMIM
from .simsiam import SimSiam
from .swav import SwAV
from .simsiam_gaze import  SimSiam_McGIP
from .moco_gaze import MoCo_McGIP
from .byol_gaze import BYOL_McGIP
from .moco_heat import MoCo_GzPT
from .byol_heat import BYOL_GzPT
from .byol_sup import BYOL_sup
from .moco_sup import MoCo_sup
__all__ = [
    'BaseModel', 'BarlowTwins', 'BYOL', 'Classification', 'DeepCluster',
    'DenseCL', 'MoCo', 'NPID', 'ODC', 'RelativeLoc', 'RotationPred', 'SimCLR',
    'SimSiam', 'SwAV', 'MAE', 'MoCoV3', 'SimMIM',
    'MMClsImageClassifierWrapper', 'CAE','SimCLR_McGIP','MoCo_McGIP','BYOL_McGIP','SimSiam_McGIP', 'MoCo_GzPT', 'BYOL_GzPT','BYOL_sup','MoCo_sup'
]
