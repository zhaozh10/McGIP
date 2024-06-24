import argparse
from modules.datasets import OAIset,trans
from modules.model import modelParser
from modules.runner import Runner
from modules.metric import metricWrapper
from utils.misc import randomSeed
import json
from torchvision import transforms as T
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
import torch
import os
def parse_agrs():
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--imgRoot', type=str, default='../data/OAI/img',
                        help='the path to the directory containing the img.')
    parser.add_argument('--dataSplit', type=str, default='../data/OAI/OAI.json',
                        help='the path to the json file containing the dataset split.')

    # Data loader settings
    parser.add_argument('--batch_size', type=int, default=32, help='the number of samples for a batch')
    parser.add_argument('--numClass', type=int, default=5)


    # Model settings
    parser.add_argument('--arch',default='resnet50')
    parser.add_argument('--frozen', type=int, default=0)
    parser.add_argument('--preTrain',default=True,help="whether load upstream pre-trained weigths")

    # log setting
    parser.add_argument('--n_gpu', type=int, default=1, help='the number of gpus to be used.')
    parser.add_argument('--epochs', type=int, default=30, help='the number of training epochs.')
    # parser.add_argument('--save_dir', type=str, default='./work_dirs/byol/', help='the path to save the models.')
    parser.add_argument('--record_dir', type=str, default='./work_dirs/byol-GzPT_resnet50_4xb32-800e_OAI/', help='the path to save the results of experiments.')
    parser.add_argument('--log_period', type=int, default=10, help='the logging interval (in batches).')
    parser.add_argument('--save_period', type=int, default=1, help='the saving period (in epochs).')
    parser.add_argument('--monitor_mode', type=str, default='max', choices=['min', 'max'], help='whether to max or min the metric.')
    parser.add_argument('--monitor_metric', type=str, default='acc', help='the metric to be monitored.')
    parser.add_argument('--early_stop', type=int, default=20, help='the patience of training.')


    # Optimization
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-5, help='the weight decay.')

    # Others
    parser.add_argument('--seed', type=int, default=1010, help='god bless you')
    parser.add_argument('--modelRoot', type=str, default='../mmselfsup/work_dirs/selfsup/byol-GzPT_resnet50_4xb32-800e_OAI/')
    parser.add_argument('--modelName', type=str, default='epoch_800.pth')
    # parser.add_argument('--modelRoot', type=str, default='../ContrastiveCrop/checkpoints/small/OAI/byol_ccrop/20230904_104746/')
    # parser.add_argument('--modelName', type=str, default='last.pth')
    

    args = parser.parse_args()
    return args

def main(args):
    
    randomSeed(args.seed)
    if  torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1
    if args.frozen:
        args.record_dir=os.path.join('./work_dirs/linearProb/',args.modelRoot.split('/')[-2]+'/')
    else:
        args.record_dir=os.path.join('./work_dirs/',args.modelRoot.split('/')[-2]+'/')

    f=open(args.dataSplit)
    splitInfo=json.load(f)

    trainSet=OAIset(dataRoot=args.imgRoot,annFile=splitInfo['train'],transforms=trans['train'])
    valSet=OAIset(dataRoot=args.imgRoot,annFile=splitInfo['val'],transforms=trans['val'])
    testSet=OAIset(dataRoot=args.imgRoot,annFile=splitInfo['test'],transforms=trans['val'])
    train_loader=DataLoader(trainSet,batch_size=16,num_workers=2,shuffle=True)
    val_loader=DataLoader(valSet,batch_size=8,num_workers=2,shuffle=False,drop_last=False)
    test_loader=DataLoader(testSet,batch_size=8,num_workers=2,shuffle=False,drop_last=False)


    net=modelParser(args)
    if(args.frozen):
        for k, v in net.named_parameters():
            if not k.startswith('fc'):
                v.requires_grad = False  # 固定参数
    criterion = nn.CrossEntropyLoss()
    metrics = metricWrapper

    # build optimizer, learning rate scheduler
    if args.frozen:
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.wd)
    else:
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)


    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)


    # build trainer and start to train
    trainer = Runner(net, criterion, metrics, optimizer, args, lr_scheduler, train_loader, val_loader, test_loader)
    trainer.train()




if __name__=="__main__":
    # parse arguments 
    args = parse_agrs()
    main(args)