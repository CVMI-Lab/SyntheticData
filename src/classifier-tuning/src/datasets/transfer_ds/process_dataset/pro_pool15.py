import glob
import os
# pwd = /opt/tiger/filter_transfer/data
new_ds = 'ds15img'
all_ds = sorted(glob.glob('pool15/*'))
os.system('mkdir {}'.format(new_ds))

for ds in all_ds:
    os.system('mv {}/train/*/* {}'.format(ds, new_ds))

