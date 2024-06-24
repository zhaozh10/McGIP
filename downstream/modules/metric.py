from optparse import Option
from typing import Optional, Dict
from torchmetrics.functional import mean_absolute_error as mae
from torchmetrics.functional import accuracy
from torch import Tensor


def metricWrapper(preds:Tensor,gt:Tensor,numClass:Optional[int]=5)-> Dict[Tensor,Tensor]:
    return {'acc':accuracy(preds, gt, num_classes=numClass,task='multiclass'),'mae':mae(preds=preds,target=gt)}
