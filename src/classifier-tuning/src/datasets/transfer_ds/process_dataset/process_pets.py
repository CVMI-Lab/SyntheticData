import glob
import os

names=[]

all = sorted(glob.glob('*'))
for i in all:
    # os.system('mv {} train/'.format(i))
    # os.system('mkdir val/{}'.format(i))

    name = sorted(glob.glob('{}/*.jpg'.format(i)))[0]

    # name = 'a '+name.split('/')[-1].split('_')[0]
    name = 'a '+name.split('/')[-1].split('_1')[0]
    # name = sorted(glob.glob('train/{}/*.jpg'.format(i)))[0].split('_')[0]

    print(name)
    names.append(name)
    print(names)
    # os.system('cp {} val/{}/'.format(to_cp_name,i))
import ipdb
ipdb.set_trace(context=20)
print(names)

