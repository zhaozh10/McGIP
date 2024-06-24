# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import numpy as np
from mmselfsup.utils import (batch_shuffle_ddp, batch_unshuffle_ddp,
                             concat_all_gather)
from ..builder import ALGORITHMS, build_backbone, build_head, build_neck
from .base import BaseModel


@ALGORITHMS.register_module()
class MoCo_heat(BaseModel):
    """MoCo.

    Implementation of `Momentum Contrast for Unsupervised Visual
    Representation Learning <https://arxiv.org/abs/1911.05722>`_.
    Part of the code is borrowed from:
    `<https://github.com/facebookresearch/moco/blob/master/moco/builder.py>`_.

    Args:
        backbone (dict): Config dict for module of backbone.
        neck (dict): Config dict for module of deep features to compact
            feature vectors. Defaults to None.
        head (dict): Config dict for module of loss functions.
            Defaults to None.
        queue_len (int): Number of negative keys maintained in the queue.
            Defaults to 65536.
        feat_dim (int): Dimension of compact feature vectors. Defaults to 128.
        momentum (float): Momentum coefficient for the momentum-updated
            encoder. Defaults to 0.999.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 queue_len=1280,
                 feat_dim=256,
                 momentum=0.999,
                 init_cfg=None,
                threshold=0.7, 
                # relation='./relation_dhash.npy',
                relation= './relation_cluster_heat.npy',
                 **kwargs):
        super(MoCo_heat, self).__init__(init_cfg)
        assert neck is not None
        self.encoder_q = nn.Sequential(
            build_backbone(backbone), build_neck(neck))
        self.encoder_k = nn.Sequential(
            build_backbone(backbone), build_neck(neck))

        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.backbone = self.encoder_q[0]
        self.neck = self.encoder_q[1]
        assert head is not None
        self.head = build_head(head)

        self.queue_len = queue_len
        self.momentum = momentum

        # create the queue
        # res = torch.randint(0, 6080, (queue_len, 1))
        self.register_buffer('queue', torch.randn(feat_dim, queue_len))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
        self.register_buffer('queue_idx', torch.randint(0, 700, (queue_len, 1),dtype=torch.long))
        # self.queue_idx=torch.zeros([1,queue_len],dtype=torch.long)
        # self.register_buffer('queue_idx_ptr', torch.zeros(1, dtype=torch.long))
        # self.warm_step=self.queue_len/batch_size
        self.threshold = threshold
        self.relation = np.load(relation)

    def _create_buffer(self, N, idx_list):
        # labels = torch.zeros([2*self.args.batch_size,2*self.args.batch_size])
        labels = torch.zeros([N,self.queue_len],dtype=torch.long)
        for i in range(N):
            idx = int(idx_list[i].item())
            for j in range(self.queue_len):
                jdx = int(self.queue_idx[j].item())
                if (i == j):
                    pass
                else:
                    sim = self.relation[idx][jdx]
                    # 千万不要改成下面这句，否则运算量会大幅增加
                    if (sim > self.threshold):
                        labels[i][j] = 1
        labels = labels.cuda()
        # mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
        # labels = labels[~mask].view(labels.shape[0], -1)
        return labels

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder."""
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + \
                           param_q.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys,idx_list):
        """Update queue."""
        # gather keys before updating queue
        keys = concat_all_gather(keys)


        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_len % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.transpose(0, 1)
        self.queue_idx[ptr:ptr + batch_size,0]=idx_list
        ptr = (ptr + batch_size) % self.queue_len  # move pointer

        self.queue_ptr[0] = ptr

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
        """Forward computation during training.

        Args:
            img (list[Tensor]): A list of input images with shape
                (N, C, H, W). Typically these should be mean centered
                and std scaled.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert isinstance(img,list)
        idx_list=idx
        im_q = img[0]
        im_k = img[1]
        # compute query features
        q = self.encoder_q(im_q)[0]  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            # update the key encoder
            self._momentum_update_key_encoder()

            # shuffle for making use of BN
            im_k, idx_unshuffle = batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)[0]  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = batch_unshuffle_ddp(k, idx_unshuffle)

        labels= self._create_buffer(q.shape[0], idx_list)
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        losses = self.head(l_pos, l_neg,labels)

        # update the queue
        self._dequeue_and_enqueue(k,idx_list)

        return losses
