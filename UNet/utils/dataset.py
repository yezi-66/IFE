import os
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import cv2
import torch
import glob
from albumentations import RandomBrightness, RandomContrast, CLAHE, RandomBrightnessContrast 


########################### Data Augmentation ###########################
class Normalize(object):
    def __init__(self, mean=[124.55, 118.90, 102.94], std=[ 56.77,  55.97,  57.50]):
        self.mean = mean
        self.std  = std
    
    def __call__(self, image, mask=None):
        mean = np.array([[self.mean]])
        std = np.array([[self.std]])
        image = (image - mean)/std
        if mask is None:
            return image
        return image, mask/255

class RandomVerticalFlip(object):
    def __call__(self, image, mask=None):
        if random.random() < 0.5:
            if mask is None:
                return image[::-1,:,:].copy()
            return image[::-1,:,:].copy(), mask[::-1, :].copy() 
        else:
            if mask is None:
                return image
            return image, mask 

class RandomHorizontalFlip(object): 
    def __call__(self, image, mask=None):
        if random.random() < 0.5:
            if mask is None:
                return image[:,::-1,:].copy()
            return image[:,::-1,:].copy(), mask[:,::-1].copy()
        else:
            if mask is None:
                return image
            return image, mask 

class RandomRotate(object):
    def rotate(self, x, random_angle, mode='image'):
        if mode == 'image':
            H, W, _ = x.shape
        else:
            H, W = x.shape
        image_change = cv2.getRotationMatrix2D((W/2, H/2), random_angle, 1)
        image_rotated = cv2.warpAffine(x, image_change, (W, H))

        return image_rotated

    def __call__(self, image, mask=None):
        if random.random() < 0.5:
            random_angle = np.random.randint(-90, 90)
            if mask is None:
                image = self.rotate(image, random_angle, 'image')
                return image
            image = self.rotate(image, random_angle, 'image')
            mask = self.rotate(mask, random_angle, 'mask')
            return image, mask
        else:
            if mask is None:
                return image
            return image, mask

class Padding(object): 
    def __call__(self, image, mask=None):
        h, w = image.shape[0], image.shape[1] 
        s = max(h, w)
        h_pad = s - h
        w_pad = s - w
        h_pad_0 = h_pad // 2
        h_pad_1 = h_pad - h_pad_0
        w_pad_0 = w_pad // 2
        w_pad_1 = w_pad - w_pad_0
        image = np.pad(image, pad_width=((h_pad_0, h_pad_1), (w_pad_0, w_pad_1), (0, 0)), mode='constant', constant_values=(0))
        if mask is None:
            return image
        else:
            mask = np.pad(mask, pad_width=((h_pad_0, h_pad_1), (w_pad_0, w_pad_1)), mode='constant', constant_values=(0))
            return image, mask 

class Aug_Compose(object):
    def __init__(self, transforms, p):
        self.transforms = transforms
        self.p = p

    def __call__(self, image):
        if (random.random() < self.p):
            for t in self.transforms:
                image = t(image=image)['image']
        return image

def do_nothing(image=None):
    img_lab = {}
    img_lab['image'] = image
    return img_lab

def enable_if(condition, obj):
    return obj if condition else do_nothing

class GrayAugmentation(object): 
    """ Transform to be used during training. 
    reference link: https://albumentations.ai/docs/api_reference/augmentations/transforms/
    """
    def __init__(self, p=0.9):
        self.augment = Aug_Compose([
            enable_if(1, RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, always_apply=False, p=0.5)), 
            enable_if(1, CLAHE(clip_limit=1.5, tile_grid_size=(8, 8), always_apply=False, p=0.5)), 
        ], p=p)

    def __call__(self, image):
        image = self.augment(image)
        return image

class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, mask=None):
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        if mask is None:
            return image
        mask  = cv2.resize( mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return image, mask 

class ToTensor(object):
    def __call__(self, image, mask=None):
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        if mask is None:
            return image
        mask  = torch.from_numpy(mask)
        return image, mask 


class MedDataset(data.Dataset):
    def __init__(self, trainsize, file, mode):
        self.trainsize = trainsize
        self.mode = mode
        if '.lst' in file or '.txt' in file:
            with open(file, 'r') as f:
                sal_image = [x.strip() for x in f.readlines() if os.path.exists(x.strip())]
                sal_mask = [i.replace('.png', '_mask.png') for i in sal_image]
        else:
            all_image = glob.glob(f"{file}/*")
            sal_image = [i for i in all_image if 'mask' not in i]
            sal_mask = [i.replace('.png', '_mask.png') for i in sal_image]
        self.images = sal_image
        self.gts = sal_mask
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.cv_normalize = Normalize([124.55, 118.90, 102.94], [56.77, 55.97, 57.50])
        self.cv_verticalflip = RandomVerticalFlip()
        self.cv_horizontalflip = RandomHorizontalFlip()
        self.cv_rotate = RandomRotate()
        self.cv_grayaug = GrayAugmentation()
        self.totensor = ToTensor()
        self.cv_pad = Padding()
        if mode == 'test':
            self.cv_resize = Resize(self.trainsize,self.trainsize)

    def __getitem__(self, index):
        name = self.images[index].split('/')[-1]
        try:
            image = cv2.imread(self.images[index])
        except:
            print(f"{self.images[index]} load error!!")

        if self.mode == 'train':
            try:
                mask  = cv2.imread(self.gts[index], 0)
            except:
                print(f"{self.gts[index]} load error!!")
            image = self.cv_grayaug(image)
            image, mask = self.cv_pad(image, mask)
            image, mask = self.cv_verticalflip(image, mask)
            image, mask = self.cv_horizontalflip(image, mask)
            image, mask = self.cv_rotate(image, mask)
            image, mask = self.cv_normalize(image, mask)
            return image, mask 
        elif self.mode == 'valid':
            try:
                mask  = cv2.imread(self.gts[index], 0)
            except:
                print(f"{self.gts[index]} load error!!")
            image, mask = self.cv_pad(image, mask)
            image, mask = self.cv_normalize(image, mask)
            return image, mask 
        else:
            shape = image.shape[:2]
            image = self.cv_pad(image)
            image = self.cv_normalize(image)
            image = self.cv_resize(image)
            image = self.totensor(image)
            return image, shape, name

    def __len__(self):
        return len(self.images)

    def collate(self, batch):
        size = self.trainsize[np.random.randint(0, 6)]
        image, mask = [list(item) for item in zip(*batch)]
        for i in range(len(batch)):
            image[i] = cv2.resize(image[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR).astype("float32") 
            mask[i]  = cv2.resize(mask[i],  dsize=(size, size), interpolation=cv2.INTER_LINEAR).astype("float32")

        image  = torch.from_numpy(np.stack(image, axis=0)).permute(0,3,1,2)
        mask   = torch.from_numpy(np.stack(mask, axis=0)).unsqueeze(1)
        return image, mask


def get_loader(batchsize, trainsize, shuffle=True, num_workers=12, pin_memory=True, file=None, mode='train'):
    dataset = MedDataset(trainsize, file, mode)
    data_loader = data.DataLoader(dataset=dataset,
                                  collate_fn=dataset.collate,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader
    
    