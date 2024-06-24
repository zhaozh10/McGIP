from optparse import Option
from subprocess import call
from typing import Optional
import torch
import os
import torch.nn as nn
from torchvision import models
import timm


def extractBackbone(state_dict,prefix: str='backbone')->callable:
    if prefix==None:
        for k in list(state_dict.keys()):
            if k.startswith('fc'):
                del state_dict[k]
        return state_dict

    for k in list(state_dict.keys()):
        if k.startswith(f'{prefix}.'):
            # print(k)
            if k.startswith('') and not k.startswith(f'{prefix}.fc'):
                # remove prefix
                state_dict[k[len(f"{prefix}."):]] = state_dict[k]
        # del掉不是backbone的部分
        del state_dict[k]
    return state_dict

def modelParser(args)->callable:
    # modelNames = sorted(name for name in models.__dict__
    #                  if name.islower() and not name.startswith("__")
    #                  and callable(models.__dict__[name]))
    model = models.__dict__[args.arch](pretrained=True)
    infeature = model.fc.in_features
    model.fc = nn.Linear(infeature, args.numClass)
    model.to(args.device)
    if not args.preTrain:
        return model
    modelRoot = args.modelRoot
    modelName = args.modelName
    print(f"load weights from {os.path.join(modelRoot,modelName)}")
    modelDict = torch.load(os.path.join(modelRoot,modelName))

    state_dict = modelDict['state_dict']
    extractBackbone(state_dict)
    # state_dict=modelDict['byol_state']
    # extractBackbone(state_dict,prefix='module.encoder_q')
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}  # 如果加载的参数包含在在现在模型中-就加载

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    if args.frozen:
        for k, v in model.named_parameters():
            if not k.startswith('fc'):
                v.requires_grad = False  # 固定参数

    return model


