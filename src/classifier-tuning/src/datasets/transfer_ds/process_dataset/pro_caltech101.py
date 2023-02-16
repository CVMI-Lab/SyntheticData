

import os
import numpy as np


# --- find train images
read_filepath = 'data/images_variant_trainval.txt'
names=[]
with open(read_filepath, 'r') as f:
    for line in f.readlines():
        line=line.strip().split(' ')
        img_name = line[0] + '.jpg'
        print(img_name)
        names.append(img_name)

# names = ["a "+n for n in names]
print(names)

for name in names:
    os.system('cp data/images/{} ../orig_pool15/aircraft/'.format(name))
    os.system('pwd')