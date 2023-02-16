from glob import glob 
# from . import constants as cs
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from os.path import join as osj
from PIL import Image
from torchvision import transforms
import os

TRAIN_TRANSFORMS = transforms.Compose([
            # transforms.Resize(32),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize(mean=mean, std=std),
        ])

TEST_TRANSFORMS = transforms.Compose([
        # transforms.Resize(32),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize(mean=mean, std=std),
    ])

class DTD(Dataset):
    def __init__(self, split="1", train=False, transform=TRAIN_TRANSFORMS):
        super().__init__()
        DTD_PATH='/opt/tiger/filter_transfer/data/dtd'
        train_path = osj(DTD_PATH, f"labels/train{split}.txt")
        val_path = osj(DTD_PATH, f"labels/val{split}.txt")
        test_path = osj(DTD_PATH, f"labels/test{split}.txt")
        if train:
            print(DTD_PATH)
            self.ims = open(train_path).readlines() + \
                            open(val_path).readlines()
        else:
            self.ims = open(test_path).readlines()
        
        self.full_ims = [osj(DTD_PATH, "images", x) for x in self.ims]
        
        pth = osj(DTD_PATH, f"labels/classes.txt")
        self.c_to_t = {x.strip(): i for i, x in enumerate(open(pth).readlines())}

        # self.transform = TRAIN_TRANSFORMS if train else TEST_TRANSFORMS
        self.transform = transform
        self.labels = [self.c_to_t[x.split("/")[0]] for x in self.ims]

    def __getitem__(self, index):
        im = Image.open(self.full_ims[index].strip())
        im = self.transform(im)
        return im, self.labels[index]

    def __len__(self):
        return len(self.ims)

if __name__ == "__main__":
    dtd = DTD(train=True)
    # import ipdb
    # ipdb.set_trace(context=20)
    target_folder = "/opt/tiger/filter_transfer/data/dtd/mix_dtd/"
    for im in dtd.full_ims:
        img = im[:-1]
        category = img.split('/')[-2]
        if not os.path.exists(target_folder+category):os.makedirs(target_folder+category)
        os.system('cp {} {}'.format(img, target_folder+category))
    a=1