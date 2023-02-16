import torch
from torchvision import datasets, transforms

# dataset = datasets.ImageFolder('/opt/tiger/filter_transfer/data/PASS_dataset/train',transform=transforms.ToTensor())
# dataset = datasets.ImageFolder('/opt/tiger/filter_transfer/data/imagenet/ILSVRC2012_img_train/train',transform=transforms.ToTensor())
dataset = datasets.ImageFolder('/opt/tiger/filter_transfer/data/PASS_dataset/train',
                               transform=transforms.Compose([transforms.Resize(256),
                             transforms.CenterCrop(224),
                             transforms.ToTensor()]))

# --------- PASS
# mean: tensor([0.4646, 0.4484, 0.4129])
# std: tensor([0.2750, 0.2689, 0.2885])

# dataset = datasets.ImageFolder('/opt/tiger/filter_transfer/data/imagenet/ILSVRC2012_img_train/train',
#                                transform=transforms.Compose([transforms.Resize(256),
#                              transforms.CenterCrop(224),
#                              transforms.ToTensor()]))

loader = torch.utils.data.DataLoader(dataset,
                         batch_size=1000,
                         num_workers=8,
                         shuffle=False)

# mean = 0.
# meansq = 0.
# i=0
# for data,_ in loader:
#     print('{}/{}'.format(i,len(loader)))
#     i+=1
#     mean = data.mean()
#     meansq = (data ** 2).mean()
#
# std = torch.sqrt(meansq - mean ** 2)
# print("mean: " + str(mean))
# print("std: " + str(std))
# print()

# mean = 0.0
# i=0
# for images, _ in loader:
#     batch_samples = images.size(0)
#     images = images.view(batch_samples, images.size(1), -1)
#     mean += images.mean(2).sum(0)
#     print('{}/{}'.format(i, len(loader)))
#     i+=1
#     print(mean / i / 1000)
# mean = mean / len(loader.dataset) / 1000

# import ipdb
# ipdb.set_trace(context=20)
# mean = torch.FloatTensor([0.485, 0.456, 0.406])
mean = torch.FloatTensor([0.4646, 0.4484, 0.4129])
var = 0.0
i=0
for images, _ in loader:
    batch_samples = images.size(0)
    images = images.view(batch_samples, images.size(1), -1)
    var += ((images - mean.unsqueeze(1))**2).sum([0,2])
    print('{}/{}'.format(i, len(loader)))
    i += 1
    print(torch.sqrt(var / (i*224*224)))
    print(torch.sqrt(var / (i*1000*224*224)))
std = torch.sqrt(var / (len(loader.dataset)*224*224))

import ipdb
ipdb.set_trace(context=20)

a=1
