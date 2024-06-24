# Copyright (c) OpenMMLab. All rights reserved.
from math import cos, pi
import torch
import torch.nn.functional as F
from mmcv.parallel import is_module_wrapper
from mmcv.runner import HOOKS, Hook


def update_box(train_set, eval_train_loader, model, len_ds, t=0.05):
    print(f'==> Start updating boxes...')
    model.eval()
    boxes = []
    for cur_iter, dict in enumerate(eval_train_loader):
        # drop_last=False
        # print(type(dict))
        images=dict['img']
        # print(images)
        images = images.cuda()
        # images = images.cuda(non_blocking=True)
        with torch.no_grad():
            feat_map = model(images)[0] # (N, C, H, W)

        N, Cf, Hf, Wf = feat_map.shape
        eval_train_map = feat_map.sum(1).view(N, -1)  # (N, Hf*Wf)
        eval_train_map = eval_train_map - eval_train_map.min(1, keepdim=True)[0]
        eval_train_map = eval_train_map / eval_train_map.max(1, keepdim=True)[0]
        eval_train_map = eval_train_map.view(N, 1, Hf, Wf)
        eval_train_map = F.interpolate(eval_train_map, size=images.shape[-2:], mode='bilinear')  # (N, 1, Hi, Wi)
        Hi, Wi = images.shape[-2:]

        for hmap in eval_train_map:
            hmap = hmap.squeeze(0)  # (Hi, Wi)

            h_filter = (hmap.max(1)[0] > t).int()
            w_filter = (hmap.max(0)[0] > t).int()

            h_min, h_max = torch.nonzero(h_filter).view(-1)[[0, -1]] / Hi  # [h_min, h_max]; 0 <= h <= 1
            w_min, w_max = torch.nonzero(w_filter).view(-1)[[0, -1]] / Wi  # [w_min, w_max]; 0 <= w <= 1
            boxes.append(torch.tensor([h_min, w_min, h_max, w_max]))

    boxes = torch.stack(boxes, dim=0).cuda()  # (num_iters, 4)
    # gather_boxes = [torch.zeros_like(boxes) for _ in range(dist.get_world_size())]
    # dist.all_gather(gather_boxes, boxes)
    # all_boxes = torch.stack(gather_boxes, dim=1).view(-1, 4)
    # all_boxes = torch.stack(boxes, dim=1).view(-1, 4)
    # all_boxes = all_boxes[:len_ds]
    # print(len_ds)
    # print(boxes.shape)
    all_boxes=boxes[:len_ds]

    train_set.boxes = all_boxes.cpu()

    # return all_boxes



# @HOOKS.register_module(name=['CCropHook', 'MomentumUpdateHook'])
@HOOKS.register_module()
class CCropUpdateHook(Hook):

    def __init__(self,t=0.1):
        self.t=t
        # self.end_momentum = end_momentum
        # self.update_interval = update_interval

    # def before_train_iter(self, runner):
    #     assert hasattr(runner.model.module, 'momentum'), \
    #         "The runner must have attribute \"momentum\" in algorithms."
    #     assert hasattr(runner.model.module, 'base_momentum'), \
    #         "The runner must have attribute \"base_momentum\" in algorithms."
    #     if self.every_n_iters(runner, self.update_interval):
    #         cur_iter = runner.iter
    #         max_iter = runner.max_iters
    #         base_m = runner.model.module.base_momentum
    #         m = self.end_momentum - (self.end_momentum - base_m) * (
    #             cos(pi * cur_iter / float(max_iter)) + 1) / 2
    #         runner.model.module.momentum = m

    def before_train_epoch(self, runner):
        if runner.mode == 'train' and (runner.epoch+1) >= 10 and (runner.epoch+1) % 10== 0:
            # 这里runner.dataset应当时Base_CCrop_runner里的ccropset
            # print(runner.model)
            update_box(runner.dataset[0],runner.val_dataloader,runner.model.module.backbone,len(runner.dataset[0]),t=self.t)


