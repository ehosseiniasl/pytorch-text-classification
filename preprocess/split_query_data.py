
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

def clean_text_v1(text):
    text = text.lower()
    text.replace(',', ' ')
    return text

filename = sys.argv[1]
savepath = os.path.dirname(filename) 


df = pandas.read_csv(open(filename, 'rt', encoding='utf-8'))

LABEL = 'Answer.information'

#if 'Q2' in filename:
#    LABEL = 'S0-Q2-device'
#elif 'Q1' in filename:
#    if 'yesno' in filename:
#        LABEL = 'S0-Q1-yesno'
#    elif 'format' in filename:
#        LABEL = 'S0-Q1-format'
#    elif 'recom' in filename:
#        LABEL = 'S0-Q1-recom'
#else:
#    LABEL = 'S0-Q3-action'
#
#if 'Q2' in LABEL:
#    label_set = [
#            'device_speaker_screen',
#            'device_camera',
#            'device__wearable',
#            'device_tablet',
#            'device_kitchen',
#            'device_speaker_no_screen']
#elif 'Q3' in LABEL:
#    label_set = [
#            'assist_action_ad',
#            'assist_action_image',
#            'assist_action_sub',
#            'assist_action_search',
#            'assist_action_video',
#            'assist_action_activate',
#            'assist_action_clarify'
#            ]
#elif 'Q1' in LABEL:
#    if 'format' in LABEL:
#        label_set = [
#                'Audio',
#                'Image',
#                'Video',
#                'Text',
#                'other'
#                ]
#
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

INPUT = ['Input.Step', 'Answer.query', 'Answer.question']

all_data = []
for _, r in df.iterrows():
    all_data.append((r[INPUT[0]], r[INPUT[1]], r[INPUT[2]], r[LABEL]))


data_cleaned = []
for txt1, txt2, txt3, l in all_data:
    if not isinstance(l, str):
        continue
    txt_cleaned1 = clean_text(txt1)
    txt_cleaned2 = clean_text(txt2)
    txt_cleaned3 = clean_text(txt3)
    l_cleaned = clean_text(l)

    data_cleaned.append((txt_cleaned1, txt_cleaned2, txt_cleaned2, l_cleaned))

read = data_cleaned

random.shuffle(read)
train= int(0.7*len(read))
val= int(0.1*len(read))
tr = read[:train]
vl = read[train:train+val]
ts = read[train+val:]

with open(os.path.join(savepath, 'train_step.csv'), 'wt') as f:
    f.write('{},{}\n'.format('step', 'information'))
    for l in tr:
        f.write('{},{}\n'.format(l[0], l[-1]))

with open(os.path.join(savepath, 'train_step_query.csv'), 'wt') as f:
    #f.write('{},{},{}\n'.format('step', 'query', 'information'))
    f.write('{},{}\n'.format('step_query', 'information'))
    for l in tr:
        f.write('{},{}\n'.format(l[0] + ' ' + l[1], l[-1]))


with open(os.path.join(savepath, 'train_step_query_question.csv'), 'wt') as f:
    f.write('{},{}\n'.format('step_query_question', 'information'))
    for l in tr:
        f.write('{},{}\n'.format(l[0] + ' ' + l[1] + ' ' + l[2], l[-1]))


with open(os.path.join(savepath, 'val_step.csv'), 'wt') as f:
    f.write('{},{}\n'.format('step', 'information'))
    for l in vl:
        f.write('{},{}\n'.format(l[0], l[-1]))

with open(os.path.join(savepath, 'val_step_query.csv'), 'wt') as f:
    f.write('{},{}\n'.format('step_query', 'information'))
    for l in vl:
        f.write('{},{}\n'.format(l[0] + ' ' + l[1], l[-1]))


with open(os.path.join(savepath, 'val_step_query_question.csv'), 'wt') as f:
    f.write('{},{}\n'.format('step_query_question', 'information'))
    for l in vl:
        f.write('{},{}\n'.format(l[0] + ' ' + l[1] + ' ' + l[2], l[-1]))



with open(os.path.join(savepath, 'test_step.csv'), 'wt') as f:
    f.write('{},{}\n'.format('step', 'information'))
    for l in ts:
        f.write('{},{}\n'.format(l[0], l[-1]))

with open(os.path.join(savepath, 'test_step_query.csv'), 'wt') as f:
    f.write('{},{}\n'.format('step_query', 'information'))
    for l in ts:
        f.write('{},{}\n'.format(l[0] + ' ' + l[1], l[-1]))


with open(os.path.join(savepath, 'test_step_query_question.csv'), 'wt') as f:
    f.write('{},{}\n'.format('step_query_question', 'information'))
    for l in ts:
        f.write('{},{}\n'.format(l[0] + ' ' + l[1] + ' ' +l[2], l[-1]))


