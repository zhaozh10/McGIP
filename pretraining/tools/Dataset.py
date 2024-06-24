from torch.utils.data import Dataset
import glob
import torch
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision.transforms import transforms
from torchvision.transforms import TrivialAugmentWide, AutoAugment, AugMix
import pandas as pd
import torch.nn as nn

class GaussianBlur(object):
    """blur a single image on CPU"""
    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img


def gazeFilter(gazeInfo,ratio=0.05):
    thresh =ratio*1024
    g = []
    x_list=[]
    y_list=[]
    x_prev, y_prev,_ = gazeInfo['seq'][0]
    t = 1
    duration=[]
    n_iter=1
    period_time=0
    for (x, y,_) in gazeInfo['seq'][1:]:
        timepoint=gazeInfo['duration'][n_iter]
        period_time += timepoint
        if abs(x - x_prev) < thresh and abs(y - y_prev) < thresh:
            t += 1
        else:
            g.append((x_prev, y_prev, 1))
            x_list.append(x_prev)
            y_list.append(y_prev)
            duration.append(period_time)
            x_prev, y_prev = x, y
            period_time=0
            t = 1
        n_iter+=1
    gazeInfo['seq']=g
    gazeInfo['duration']=duration
    gazeInfo['x']=x_list
    gazeInfo['y']=y_list
    return gazeInfo


def retrieve_gaze(csv_name, roi):
    df = pd.read_csv('labelme_cut_gaze/' + csv_name)
    # df_new=df.sort_values(['system_time_in_pc','img_gaze_X','img_gaze_Y'], ascending=[True,True,True])
    sum = 0
    gaze_list = []
    gaze_time = []
    x_list = []
    y_list = []
    for index, row in df.iterrows():
        x = row['img_gaze_X']
        y = row['img_gaze_Y']
        if (x < roi[2] and x > roi[0] and y > roi[1] and y < roi[3]):
            sum += 1
            gaze_time.append(row['system_time_in_pc'])
            gaze_list.append([x - roi[0], y - roi[1], 1])
            x_list.append(x - roi[0])
            y_list.append(y - roi[1])
            # print(row['system_time_in_pc'], row["img_gaze_X"], row["img_gaze_Y"])
    for step in range(len(gaze_time) - 1, 0, -1):
        gaze_time[step] = gaze_time[step] - gaze_time[step - 1]
    gaze_time = gaze_time[1:]
    gaze_list = gaze_list[1:]
    x_list = x_list[1:]
    y_list = y_list[1:]
    gazeInfo={"duration":gaze_time,"seq":gaze_list,"x":x_list,"y":y_list}
    return gazeInfo



def extract_info(root, idx):
    namelist = glob.glob(f'{root}/*.jpg')
    item_name = namelist[idx]
    # namelist[0].split('_')[0]
    elements = item_name.split('_')
    csv_name = elements[0]

    # vision = int(elements[1])
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
    roi = roi.astype(np.int)
    name={"img":item_name,"csv":csv_name}
    return label,roi,name

def train_eval_Mammo():
    namelist=glob.glob('train/*.jpg')
    train,eval=train_test_split(namelist,test_size=0.2,train_size=0.8)
    trainset=MammoSet('train',train)
    valset=MammoSet('train',eval)
    return trainset,valset

class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        tobe=[self.base_transform(x) for i in range(self.n_views)]
        return tobe


class MammoSet_v2(Dataset):
    def __init__(self,root,namelist):
        self.root=root
        self.namelist=namelist

    @staticmethod
    def get_simclr_pipeline_transform(size, s=224):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        # color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([transforms.ToPILImage(),
                                              transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                              GaussianBlur(kernel_size=int(0.1 * size)),
                                              transforms.ToTensor()])
        return data_transforms


    def __getitem__(self, idx):
        item_name=self.namelist[idx]
        img=cv2.imread(item_name,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (224, 224))
        # img = img.astype(np.float32) / 255


        img=np.expand_dims(img,axis=2)
        img=np.tile(img,(1,1,3))
        Generator=ContrastiveLearningViewGenerator(self.get_simclr_pipeline_transform(224),n_views=2)
        imgpair=Generator(img)


        return imgpair,idx


    def __len__(self):
        return len(self.namelist)

class MammoCLRSet(Dataset):
    def __init__(self,root,namelist):
        self.root=root
        self.namelist=namelist

    @staticmethod
    def get_simclr_pipeline_transform(size, s=224):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        # color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([transforms.ToPILImage(),
                                              transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                   std=[0.229, 0.224, 0.225]),
                                              GaussianBlur(kernel_size=int(0.1 * size)),
                                              transforms.ToTensor()])
        return data_transforms


    def __getitem__(self, idx):
        item_name=self.namelist[idx]
        img=cv2.imread(item_name,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (224, 224))
        # img = img.astype(np.float32) / 255
        elements=item_name.split('_')
        [h,w]=img.shape

        label,roi,name=extract_info(self.root,idx)
        # gazeInfo = retrieve_gaze(name['csv'], roi)
        img=np.expand_dims(img,axis=2)
        img=np.tile(img,(1,1,3))
        Generator=ContrastiveLearningViewGenerator(self.get_simclr_pipeline_transform(224),n_views=2)
        imgpair=Generator(img)
        record={'name':name,'roi':roi}

        return imgpair, record


    def __len__(self):
        return len(self.namelist)


class MammoSet(Dataset):
    def __init__(self,root,namelist):
        self.root=root
        # if(root=='train'):
        #     train1, train2 = sklearn.model_selection.train_test_split(namelist, train_size=0.2, test_size=0.8)
        #     namelist=train1
        self.namelist=namelist
        # self.namelist=glob.glob(f'{self.root}/*.jpg')
        # self.namelist = glob.glob(f'train/*.jpg')

    def __getitem__(self, idx):
        size=224
        item_name=self.namelist[idx]
        img=cv2.imread(item_name,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255
        elements=item_name.split('_')
        [h,w]=img.shape
        # vision = int(elements[1])
        label=-1
        roi=np.zeros([4,1])
        if(self.root=='train'):
            if (len(elements)==6 or len(elements)==7):
                label=int(elements[-4][-1])
                roi = np.array(elements[-2][3:].split('-'))
            elif (len(elements)==8 or len(elements)==9):
                label=int(elements[-6][-1])
                roi = np.array(elements[-3][3:].split('-'))
        else:
            if(len(elements)==6):
                label = int(elements[-4][-1])
                roi = np.array(elements[-3][3:].split('-'))
            else:
                label = int(elements[-2][-1])
                roi = np.array(elements[-1][3:].split('-'))
        # return self.root,self.namelist
        img=np.expand_dims(img,axis=0)
        img=np.tile(img,(3,1,1))
        img=torch.from_numpy(img)

        # only for augmix()
        # img=(img*255).to(torch.uint8)

        data_transforms = transforms.Compose([

                                              # transforms.ToPILImage(),

                                              # transforms.RandomResizedCrop(size=224),
                                              # transforms.RandomHorizontalFlip(),

                                            # TrivialAugmentWide(),
                                            # AutoAugment(),
                                            # AugMix(),
                                            # v2.ConvertDtype(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                   std=[0.229, 0.224, 0.225]),
                                              # GaussianBlur(kernel_size=int(0.1 * size)),
                                                # transforms.ToTensor(),
                                              ])
        # img=torch.tensor(img*255,dtype=torch.uint8)
        img=data_transforms(img)
        # norm=transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        # img=norm(img/255)
        
        return img, label
        # return img,label,roi,vision

    def get_img_idx(self, idx):
        size=224
        item_name=self.namelist[idx]
        img=cv2.imread(item_name,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (size, size))
        img = img.astype(np.float32) / 255
        elements=item_name[:-4].split('_')
        [h,w]=img.shape
        # vision = int(elements[1])
        label=-1
        roi=np.zeros([4,1])
        dest=np.zeros([4,1])
        if(self.root=='train'):
            if (len(elements)==6 or len(elements)==7):
                label=int(elements[-4][-1])
                roi = np.array(elements[-2][3:].split('-'))
            elif (len(elements)==8 or len(elements)==9):
                label=int(elements[-6][-1])
                roi = np.array(elements[-3][3:].split('-'))
        else:
            if(len(elements)==6):
                label = int(elements[-4][-1])
                roi = np.array(elements[-3][3:].split('-'))
                dest = np.array(elements[-1][4:].split('-'))
            else:
                label = int(elements[-2][-1])
                roi = np.array(elements[-1][3:].split('-'))
                dest = np.array(elements[-1][4:].split('-'))
        # return self.root,self.namelist
        img=np.expand_dims(img,axis=0)
        img=np.tile(img,(3,1,1))
        img=torch.from_numpy(img)
        data_transforms = transforms.Compose([
                                              # transforms.ToPILImage(),

                                              # transforms.RandomResizedCrop(size=224),
                                              # transforms.RandomHorizontalFlip(),
                                        
                                            #   TrivialAugmentWide(), 
                                                # AutoAug(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                   std=[0.229, 0.224, 0.225]),
                                              # GaussianBlur(kernel_size=int(0.1 * size)),
                                                # transforms.ToTensor(),
                                              ])
        img=data_transforms(img)
        if(label==2):
            return img,label,roi,dest
        return img, label
        # return img,label,roi,vision

    def __len__(self):
        return len(self.namelist)


class MammoSet_hr(Dataset):
    def __init__(self,root,namelist):
        self.root=root
        # if(root=='train'):
        #     train1, train2 = sklearn.model_selection.train_test_split(namelist, train_size=0.2, test_size=0.8)
        #     namelist=train1
        self.namelist=namelist
        # self.namelist=glob.glob(f'{self.root}/*.jpg')
        # self.namelist = glob.glob(f'train/*.jpg')

    def __getitem__(self, idx):
        size=224
        item_name=self.namelist[idx]
        img=cv2.imread(item_name,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255
        elements=item_name.split('_')
        [h,w]=img.shape
        # vision = int(elements[1])
        label=-1
        roi=np.zeros([4,1])
        if(self.root=='train'):
            if (len(elements)==6 or len(elements)==7):
                label=int(elements[-4][-1])
                roi = np.array(elements[-2][3:].split('-'))
            elif (len(elements)==8 or len(elements)==9):
                label=int(elements[-6][-1])
                roi = np.array(elements[-3][3:].split('-'))
        else:
            if(len(elements)==6):
                label = int(elements[-4][-1])
                roi = np.array(elements[-3][3:].split('-'))
            else:
                label = int(elements[-2][-1])
                roi = np.array(elements[-1][3:].split('-'))
        # return self.root,self.namelist
        img=np.expand_dims(img,axis=0)
        img=np.tile(img,(3,1,1))
        img=torch.from_numpy(img)
        data_transforms = transforms.Compose([
                                              # transforms.ToPILImage(),

                                              # transforms.RandomResizedCrop(size=224),
                                              # transforms.RandomHorizontalFlip(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                   std=[0.229, 0.224, 0.225]),
                                              # GaussianBlur(kernel_size=int(0.1 * size)),
                                                # transforms.ToTensor(),
                                              ])
        img=data_transforms(img)
        return img, label
        # return img,label,roi,vision

    def get_img_idx(self, idx):
        size=1024
        item_name=self.namelist[idx]
        img=cv2.imread(item_name,cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, (size, size))
        img = img.astype(np.float32) / 255
        elements=item_name[:-4].split('_')
        [h,w]=img.shape
        # vision = int(elements[1])
        label=-1
        roi=np.zeros([4,1])
        dest=np.zeros([4,1])
        if(self.root=='train'):
            if (len(elements)==6 or len(elements)==7):
                label=int(elements[-4][-1])
                roi = np.array(elements[-2][3:].split('-'))
            elif (len(elements)==8 or len(elements)==9):
                label=int(elements[-6][-1])
                roi = np.array(elements[-3][3:].split('-'))
        else:
            if(len(elements)==6):
                label = int(elements[-4][-1])
                roi = np.array(elements[-3][3:].split('-'))
                dest = np.array(elements[-1][4:].split('-'))
            else:
                label = int(elements[-2][-1])
                roi = np.array(elements[-1][3:].split('-'))
                dest = np.array(elements[-1][4:].split('-'))
        # return self.root,self.namelist
        img=np.expand_dims(img,axis=0)
        img=np.tile(img,(3,1,1))
        img=torch.from_numpy(img)
        data_transforms = transforms.Compose([
                                              # transforms.ToPILImage(),

                                              # transforms.RandomResizedCrop(size=224),
                                              # transforms.RandomHorizontalFlip(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                   std=[0.229, 0.224, 0.225]),
                                              # GaussianBlur(kernel_size=int(0.1 * size)),
                                                # transforms.ToTensor(),
                                              ])
        img=data_transforms(img)
        if(label==2):
            return img,label,roi,dest
        return img, label
        # return img,label,roi,vision

    def __len__(self):
        return len(self.namelist)

if __name__ == "__main__":

    root='test'
    namelist=glob.glob(f'{root}/*.jpg')

    # for name in namelist:
    #     elements=name.split('_')
    #     if (len(elements) == 6):
    #         label = int(elements[-4][-1])
    #         roi = np.array(elements[-3][3:].split('-'))
    #     else:
    #         label = int(elements[-2][-1])
    #         roi = np.array(elements[-1][3:].split('-'))

    # trainset=MammoSet(root)
    # trainloader=DataLoader(trainset,batch_size=16,shuffle=True,pin_memory=True)
    # img,label=next(iter(trainloader))
    # img, label, roi, vision = next(iter(trainloader))
    # sum=0
    # for name in namelist:
    #     elements=name.split('_')
    #     if(len(elements)!=6 and len(elements)!=8):
    #         sum+=1
    #         print(len(elements))
    #         print(name)
    print(0)


