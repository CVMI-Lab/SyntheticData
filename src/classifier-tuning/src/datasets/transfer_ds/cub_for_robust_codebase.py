import os
import cv2
import numpy as np
from torch.utils.data import Dataset

def main():
    path= "/opt/tiger/filter_transfer/data/CUB_200_2011"
    root = path
    images_path = {}

    os.system('mkdir train')
    os.system('cp -r images/* train/')
    os.system('mkdir val')
    os.system('cp -r images/* val/')

    with open(os.path.join(root, 'images.txt')) as f:
        for line in f:
            image_id, path = line.split()
            images_path[image_id] = path

    class_ids = {}
    with open(os.path.join(root, 'image_class_labels.txt')) as f:
        for line in f:
            image_id, class_id = line.split()
            class_ids[image_id] = class_id

    train_id = []   # train not val
    with open(os.path.join(root, 'train_test_split.txt')) as f:
        for line in f:
            image_id, is_train = line.split()
            if int(is_train):
                train_id.append(image_id)

    with open(os.path.join(root, 'images.txt')) as f:
        for line in f:
            image_id, path = line.split()
            if image_id in train_id:
                os.system('rm val/{}'.format(path))
            else:
                # import ipdb
                # ipdb.set_trace(context=20)
                os.system('rm train/{}'.format(path))






if __name__ == "__main__":
    main()