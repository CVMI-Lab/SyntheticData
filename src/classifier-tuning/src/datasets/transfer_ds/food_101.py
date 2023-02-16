# pytorch imports
import torch
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from robustness import data_augmentation as da
from . import constants as cs

class FOOD101():
    def __init__(self, transform=None):
        # self.TRAIN_PATH = cs.FOOD_PATH+"/train"
        self.TRAIN_PATH = "/opt/tiger/filter_transfer/data/food-101/train"
        # self.VALID_PATH = cs.FOOD_PATH+"/valid"
        self.VALID_PATH = "/opt/tiger/filter_transfer/data/food-101/valid"

        self.train_ds, self.valid_ds, self.train_cls, self.valid_cls = [None]*4
        self.transform = transform
   
    def _get_tfms(self):
        train_tfms = cs.TRAIN_TRANSFORMS
        valid_tfms = cs.TEST_TRANSFORMS
        return train_tfms, valid_tfms            
            
    def get_dataset(self):
        # train_tfms, valid_tfms = self._get_tfms() # transformations
        train_tfms, valid_tfms = self.transform, self.transform
        self.train_ds = datasets.ImageFolder(root=self.TRAIN_PATH,
                                        transform=train_tfms)
        self.valid_ds = datasets.ImageFolder(root=self.VALID_PATH,
                                        transform=valid_tfms)        
        self.train_classes = self.train_ds.classes
        self.valid_classes = self.valid_ds.classes

        # print(self.train_classes)

        assert self.train_classes==self.valid_classes
        return self.train_ds, self.valid_ds, self.train_classes
    
    def get_dls(self, train_ds, valid_ds, bs, **kwargs):
        return (DataLoader(train_ds, batch_size=bs, shuffle=True, **kwargs),
               DataLoader(valid_ds, batch_size=bs, shuffle=True, **kwargs))
   


