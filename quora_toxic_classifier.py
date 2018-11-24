import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from nltk import word_tokenize
from nltk.corpus import stopwords
import re
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from nltk.stem import SnowballStemmer, WordNetLemmatizer
import time
import pickle
import bcolz
import tensorflow as tf

dataset_train = pd.read_csv('dataset/train.csv')
dataset_test = pd.read_csv('dataset/test.csv')

print(os.listdir('./dataset'))

print(dataset_train.groupby(['target']).size())

stop = set(stopwords.words('english'))

print(dataset_train.head())
print(dataset_train.shape)
print(dataset_test.head())
print(dataset_test.shape)

dataset_train = dataset_train.drop(['qid'],axis=1)
dataset_test = dataset_test.drop(['qid'],axis=1)

stemmer = SnowballStemmer('english')

def lemmatize(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text))

starttime = time.time()
for idx,row in dataset_train.iterrows():
    nval = ''
    for val in row['question_text'].split(' '):
        val = re.sub('[^A-Za-z]+',' ',val)
        if(val != ' '):
            val = lemmatize(val)
            if val.lower() not in stop:
                nval = nval + ' ' + val.lower()
    dataset_train.at[idx,'question_text'] = nval
    if(idx%100000 == 0):
        if(idx != 0):
            endtime = time.time()
            print(endtime - starttime)
            starttime = endtime
        print(idx)
        
starttime= time.time()
for idx,row in dataset_test.iterrows():
    nval = ''
    for val in row['question_text'].split(' '):
        val = re.sub('[^A-Za-z]+',' ',val)
        if(val != ' '):
            val = lemmatize(val)
            if val.lower() not in stop:
                nval = nval + ' ' + val.lower()
    dataset_test.at[idx,'question_text'] = nval
    if(idx%100000 == 0):
        if(idx != 0):
            endtime = time.time()
            print(endtime - startime)
            starttime = endtime
        print(idx)
        
print(dataset_train.head())
print(dataset_test.head())

dataset_train.to_csv('./dataset/processed_train.csv',sep=',')
dataset_test.to_csv('./dataset/processed_test.csv',sep=',')

dataset_train = pd.read_csv('./dataset/processed_train.csv')
    
X_train,X_test,y_train,y_test = train_test_split(dataset_train.iloc[:,0],dataset_train.iloc[:,1],test_size=0.20,random_state=0)

for i in range(0,len(X_train)):
    
print(X_train.shape)
print(X_test.shape)


