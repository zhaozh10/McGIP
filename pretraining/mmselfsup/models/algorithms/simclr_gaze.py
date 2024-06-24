# Copyright (c) OpenMMLab. All rights reserved.
import torch
import numpy as np
from ..builder import ALGORITHMS, build_backbone, build_head, build_neck
from ..utils import GatherLayer
from .base import BaseModel


@ALGORITHMS.register_module()
class SimCLR_gaze(BaseModel):
    """SimCLR.

    Implementation of `A Simple Framework for Contrastive Learning
    of Visual Representations <https://arxiv.org/abs/2002.05709>`_.

    Args:
        backbone (dict): Config dict for module of backbone.
        neck (dict): Config dict for module of deep features to compact feature
            vectors. Defaults to None.
        head (dict): Config dict for module of loss functions.
            Defaults to None.
    """

    def __init__(self, backbone, neck=None, head=None, init_cfg=None,weights='0.1-0.1-0.1-0.1-0.6',threshold=0.5,relation='/home/zihao/mmself/relation.npy'):
        super(SimCLR_gaze, self).__init__(init_cfg)
        self.backbone = build_backbone(backbone)
        self.weights=weights
        self.threshold=threshold
        self.relation=np.load(relation)
        assert neck is not None
        self.neck = build_neck(neck)
        assert head is not None
        self.head = build_head(head)

    def _create_buffer(self,N,idx_list):
        weights=np.array(self.weights.split('-')).astype(float)
        # labels = torch.zeros([2*self.args.batch_size,2*self.args.batch_size])
        labels = torch.cat([torch.arange(N) for i in range(2)], dim=0)
        
        # 此时labels有2*batchsize行，每行有2个1，其余为0
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        for i in range(N):
            idx=int(idx_list[i].item())
            for j in range(i,N):
                jdx = int(idx_list[j].item())
                if(i==j):
                    pass
                else:
                    # sim = m.docomparison(gazeInfo[i], gazeInfo[j], screensize=[1024, 1024],grouping=True,TDir=45,TDur=2000,TAmp=120)
                    sim = self.relation[idx][jdx]
                    # 千万不要改成下面这句，否则运算量会大幅增加
                    # sim = m.docomparison(gazeInfo[i], gazeInfo[j], screensize=[1024, 1024], grouping=False)
                    sim = np.array(sim)
                    score = np.dot(sim, weights)
                    if(score>self.threshold):
                        labels[i][j]=1
                        labels[j][i]=1
        labels=labels.cuda()
        mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
        labels = labels[~mask].view(labels.shape[0], -1)
        return labels,mask

    def extract_feat(self, img):
        """Function to extract features from backbone.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            tuple[Tensor]: backbone outputs.
        """
        x = self.backbone(img)
        return x

    def forward_train(self, img,idx, **kwargs):
        # Assume that idx has been successfully input, which remains to be solved
        """Forward computation during training.
        # the img here is a batch of images, and ,maybe **kwargs can return some other info?
        Args:
            img (list[Tensor]): A list of input images with shape
                (N, C, H, W). Typically these should be mean centered
                and std scaled.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # idx_list=kwargs['idx']
        idx_list=idx
        # print(type(idx_list))
        # print(idx_list)
        assert isinstance(img, list)
        # assert isinstance(idx, list)
        # np.asarray(idx_list.cpu(),dtype=int)
        img = torch.concat(img, 0)
        x = self.extract_feat(img)  # 2n
        out=self.neck(x)
        # print("==================")
        z = self.neck(x)[0]  # (2n)xd
        z = z / (torch.norm(z, p=2, dim=1, keepdim=True) + 1e-10)
        z = torch.cat(GatherLayer.apply(z), dim=0)  # (2N)xd
        assert z.size(0) % 2 == 0
        N = z.size(0) // 2
        s = torch.matmul(z, z.permute(1, 0))  # (2N)x(2N)
        labels,mask= self._create_buffer(N, idx_list)
        # print(z.shape)
        # print(mask.shape)
        # print(labels.shape)
        s = s[~mask].view(z.shape[0], -1)
        # mask, pos_ind, neg_mask = self._create_buffer(N,idx_list)
        # remove diagonal, (2N)x(2N-1)
        # s = torch.masked_select(s, mask == 1).reshape(s.size(0), -1)
        # positive = s[pos_ind].unsqueeze(1)  # (2N)x1
        # # select negative, (2N)x(2N-2)
        # negative = torch.masked_select(s, neg_mask == 1).reshape(s.size(0), -1)
        losses = self.head(s,labels)
        # losses = self.head(positive, negative)
        return losses
