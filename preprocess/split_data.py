
import os
import sys
import random
import ipdb
import csv
import re

filename = sys.argv[1]
savepath = os.path.dirname(filename) 

reader =list(csv.reader(open(filename,'rt')))
read = reader[1:]
random.shuffle(read)


train= int(0.7*len(read))
val= int(0.1*len(read))
tr = read[:train]
vl = read[train:train+val]
ts = read[train+val:]

with open(os.path.join(savepath, 'train.csv'), 'wt') as f:
    f.write('{},{}\n'.format('recipedetails', 'label'))
    for l in tr:
        if 'S0' in l[1]:
            continue
        txt = re.sub(r'[^\w\s]','',l[0]).lower()
        if txt in ['', ' ']:
            continue
        f.write('{},{}\n'.format(txt,l[-1]))


with open(os.path.join(savepath, 'val.csv'), 'wt') as f:
    f.write('{},{}\n'.format('recipedetails', 'label'))
    for l in vl:
        if 'S0' in l[1]:
            continue
        txt = re.sub(r'[^\w\s]','',l[0]).lower()
        if txt in ['', ' ']:
            continue
        f.write('{},{}\n'.format(txt,l[-1]))


with open(os.path.join(savepath, 'test.csv'), 'wt') as f:
    f.write('{},{}\n'.format('recipedetails', 'label'))
    for l in ts:
        if 'S0' in l[1]:
            continue
        txt = re.sub(r'[^\w\s]','',l[0]).lower()
        if txt in ['', ' ']:
            continue
        f.write('{},{}\n'.format(txt,l[-1]))
