import glob
import os

src_classes = sorted(glob.glob('ILSVRC2012_img_train/train/*'))
os.system('mkdir -p ILSVRC2012_img_train_plus15tasks/train/img')

i=0
for src_cls in src_classes:
    print(i)
    i+=1
    os.system('cp {}/* ILSVRC2012_img_train_plus15tasks/train/img'.format(src_cls))

