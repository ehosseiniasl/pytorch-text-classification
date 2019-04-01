
import pandas
import math
import os
import sys
import random
import ipdb
import csv
import re
import nltk
from nltk.stem import PorterStemmer
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = stopwords.words('english')
porter = PorterStemmer()

def tokenizer_porter(text):
    return ' '.join([porter.stem(word) for word in text.split()])

def remove_stop_words(text):
    return ' '.join([word for word in text.split() if word not in stop_words])

def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    text = tokenizer_porter(text)
    text = remove_stop_words(text)
    return text

filename = sys.argv[1]
savepath = os.path.dirname(filename) 


df = pandas.read_csv(open(filename, 'rt'))

if 'Q2' in filename:
    LABEL = 'S0-Q2-device'
elif 'Q1' in filename:
    if 'yesno' in filename:
        LABEL = 'S0-Q1-yesno'
    elif 'format' in filename:
        LABEL = 'S0-Q1-format'
    elif 'recom' in filename:
        LABEL = 'S0-Q1-recom'
else:
    LABEL = 'S0-Q3-action'

if 'Q2' in LABEL:
    label_set = [
            'device_speaker_screen',
            'device_camera',
            'device__wearable',
            'device_tablet',
            'device_kitchen',
            'device_speaker_no_screen']
elif 'Q3' in LABEL:
    label_set = [
            'assist_action_ad',
            'assist_action_image',
            'assist_action_sub',
            'assist_action_search',
            'assist_action_video',
            'assist_action_activate',
            'assist_action_clarify'
            ]
elif 'Q1' in LABEL:
    if 'format' in LABEL:
        label_set = [
                'Audio',
                'Image',
                'Video',
                'Text',
                'other'
                ]

#label_set = []
#labels = df[LABEL]
#for l in labels:
#    if not isinstance(l, str):
#        continue
#    tmp = l.strip().split('$')
#    for t in tmp:
#        if t not in label_set:
#            label_set.append(t)
#label_set.remove('')
#label_set.sort()

all_data = []
for _, r in df.iterrows():
    all_data.append((r['recipedetails'], r[LABEL]))


data_cleaned = []
for txt, l in all_data:
    if not isinstance(l, str):
        continue
    txt_cleaned = clean_text(txt)
    
    if LABEL == 'S0-Q1-yesno':
        data_cleaned.append((txt_cleaned, l))
    else:
        labels = l.strip().split('$')
        labels.remove('')
        if labels == None:
            continue
        label_ids = [0]*len(label_set)
        for l in labels:
            label_ids[label_set.index(l)] = 1

        data_cleaned.append((txt_cleaned, label_ids))

read = data_cleaned

random.shuffle(read)
train= int(0.7*len(read))
val= int(0.1*len(read))
tr = read[:train]
vl = read[train:train+val]
ts = read[train+val:]

with open(os.path.join(savepath, 'train.csv'), 'wt') as f:
    if LABEL != 'S0-Q1-yesno':
        f.write('{},{}\n'.format('recipedetails', ','.join(label_set)))
        for l in tr:
            if l[0] in ['', ' ']:
                continue
            f.write('{},{}\n'.format(l[0], ','.join(map(str, l[1]))))
    else:
        f.write('{},{}\n'.format('recipedetails', 'label'))
        for l in tr:
            if l[0] in ['', ' ']:
                continue
            f.write('{},{}\n'.format(l[0], l[1]))


with open(os.path.join(savepath, 'val.csv'), 'wt') as f:
    if LABEL != 'S0-Q1-yesno':
        f.write('{},{}\n'.format('recipedetails', ','.join(label_set)))
        for l in vl:
            if l[0] in ['', ' ']:
                continue
            f.write('{},{}\n'.format(l[0], ','.join(map(str, l[1]))))
    else:
        f.write('{},{}\n'.format('recipedetails', 'label'))
        for l in vl:
            if l[0] in ['', ' ']:
                continue
            f.write('{},{}\n'.format(l[0], l[1]))


with open(os.path.join(savepath, 'test.csv'), 'wt') as f:
    if LABEL != 'S0-Q1-yesno':
        f.write('{},{}\n'.format('recipedetails', ','.join(label_set)))
        for l in ts:
            if l[0] in ['', ' ']:
                continue
            f.write('{},{}\n'.format(l[0], ','.join(map(str, l[1]))))
    else:
        f.write('{},{}\n'.format('recipedetails', 'label'))
        for l in ts:
            if l[0] in ['', ' ']:
                continue
            f.write('{},{}\n'.format(l[0], l[1]))


with open(os.path.join(savepath, 'trainval.csv'), 'wt') as f:
    if LABEL != 'S0-Q1-yesno':
        f.write('{},{}\n'.format('recipedetails', ','.join(label_set)))
        for l in tr+vl:
            if l[0] in ['', ' ']:
                continue
            f.write('{},{}\n'.format(l[0], ','.join(map(str, l[1]))))
    else:
        f.write('{},{}\n'.format('recipedetails', 'label'))
        for l in tr+vl:
            if l[0] in ['', ' ']:
                continue
            f.write('{},{}\n'.format(l[0], l[1]))
