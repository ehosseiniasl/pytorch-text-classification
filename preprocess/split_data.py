
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
        #txt = re.sub(r'[^\w\s]','',l[0]).lower()
        txt = clean_text(l[0])
        if txt in ['', ' ']:
            continue
        f.write('{},{}\n'.format(txt,l[-1]))


with open(os.path.join(savepath, 'val.csv'), 'wt') as f:
    f.write('{},{}\n'.format('recipedetails', 'label'))
    for l in vl:
        if 'S0' in l[1]:
            continue
        #txt = re.sub(r'[^\w\s]','',l[0]).lower()
        txt = clean_text(l[0])
        if txt in ['', ' ']:
            continue
        f.write('{},{}\n'.format(txt,l[-1]))


with open(os.path.join(savepath, 'test.csv'), 'wt') as f:
    f.write('{},{}\n'.format('recipedetails', 'label'))
    for l in ts:
        if 'S0' in l[1]:
            continue
        #txt = re.sub(r'[^\w\s]','',l[0]).lower()
        txt = clean_text(l[0])
        if txt in ['', ' ']:
            continue
        f.write('{},{}\n'.format(txt,l[-1]))
