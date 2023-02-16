from torchvision import transforms
from .Randaug import RandAugment

prefix = "data"

# IMGNET_PATH = prefix + "/imagenet/ILSVRC2012_img_train_100k/"
# IMGNET_PATH = prefix + "/imagenet/ILSVRC2012_img_train_128k/"
# IMGNET_PATH = prefix + "/imagenet/ILSVRC2012_img_train_200k/"
IMGNET_PATH = prefix + "/imagenet/ILSVRC2012_img_train/"
# IMGNET_PATH = prefix + "/imagenet/ILSVRC2012_img_train_plus15tasks/"
# IMGNET_PATH = prefix + "/imagenet/imagenet.10_1000/"
# IMGNET_PATH = prefix + "/imagenet/ILSVRC2012_img_traintrain_200cls_640shot_128.0k/"
# IMGNET_PATH = prefix + "/imagenet/ILSVRC2012_img_traintrain_500cls_256shot_128.0k/"
# IMGNET_PATH = prefix + "/place_128k"
# IMGNET_PATH = prefix + "/pass/PASS_128k"

# Planes dataset
FGVC_PATH = prefix + "/fgvc-aircraft-2013b/"

# Oxford Flowers dataset
# FLOWERS_PATH = prefix + "/oxford_flowers_pytorch/"
FLOWERS_PATH = prefix + "/flowers_new/"

# DTD dataset
DTD_PATH = prefix + "/dtd/"

# Stanford Cars dataset
CARS_PATH = prefix + "/cars_new"

# SUN397 dataset
SUN_PATH = prefix + "/SUN397/splits_01/"

# FOOD dataset
FOOD_PATH = prefix + "/food-101"

# BIRDS dataset
BIRDS_PATH = prefix + "/birdsnap"

# CUB-200-2011 birds
CUB_PATH = prefix + "/CUB_200_2011"

# COCO
COCO_PATH = prefix + "/coco_cls"

# ade20k
ADE20K_PATH = prefix + "/ade20k_cls"

# Mix seg: cs voc ade
MIX_SEG_PATH = prefix + "/mix_seg"

# PETS dataset
PETS_PATH = prefix + ""

# Caltech datasets
CALTECH101_PATH = prefix + ""
CALTECH256_PATH = prefix + ""

value_scale = 255
mean = [0.485, 0.456, 0.406]
mean = [0.48145466, 0.4578275, 0.40821073]
mean = [item * value_scale for item in mean]
std = [0.229, 0.224, 0.225]
std = [0.26862954, 0.26130258, 0.27577711]
std = [item * value_scale for item in std]

# Data Augmentation defaults
TRAIN_TRANSFORMS = transforms.Compose([
            # transforms.Resize(32),
            transforms.RandomResizedCrop(224),
            # transforms.RandomResizedCrop(224, scale=(0.08,1.0), ratio=(0.75,1.333333)),
            # transforms.RandomResizedCrop(224, scale=(0.08,1.0), ratio=(0.5,2.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize(mean=mean, std=std),
        ])
# TRAIN_TRANSFORMS = transforms.Compose([
#         # transforms.Resize(32),
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         # transforms.Normalize(mean=mean, std=std),
#     ])

TEST_TRANSFORMS = transforms.Compose([
        # transforms.Resize(32),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize(mean=mean, std=std),
    ])

# from PIL import Image
# BICUBIC = Image.BICUBIC
# TEST_TRANSFORMS = transforms.Compose([
#         # transforms.Resize(32),
#         transforms.Resize(224,interpolation=BICUBIC),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         # transforms.Normalize(mean=mean, std=std),
#     ])

# Add RandAugment with N, M(hyperparameter)
# N=3
# M=9
# TRAIN_TRANSFORMS.transforms.insert(0, RandAugment(N, M))