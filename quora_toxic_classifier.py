import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf

dataset_train = pd.read_csv('dataset/train.csv')
dataset_test = pd.read_csv('dataset/test.csv')

print(dataset_train.groupby(['target']).size())

stop = set(stopwords.words('english'))

print(dataset_train.head())
print(dataset_train.shape)
print(dataset_test.head())
print(dataset_test.shape)

dataset_train = dataset_train.drop(['qid'],axis=1)
dataset_test = dataset_test.drop(['qid'],axis=1)

ps = PorterStemmer()
for idx,row in dataset_train.iterrows():
    nval = ''
    for val in row['question_text'].split(' '):
        val = re.sub('[^A-Za-z]+',' ',val)
        if(val != ' '):
            val = ps.stem(val)
            if val.lower() not in stop:
                nval = nval + ' ' + val.lower()
    dataset_train.at[idx,'question_text'] = nval
for idx,row in dataset_test.iterrows():
    nval = ''
    for val in row['question_text'].split(' '):
        val = re.sub('[^A-Za-z]+',' ',val)
        if(val != ' '):
            val = ps.stem(val)
            if val.lower() not in stop:
                nval = nval + ' ' + val.lower()
    dataset_test.at[idx,'question_text'] = nval
print(dataset_train.head())
print(dataset_test.head())

dataset_train.to_csv('./dataset/processed_train.csv',sep=',')

dataset_train = pd.read_csv('./dataset/processed_train.csv')

X_train,X_test,y_train,y_test = train_test_split(dataset_train.iloc[:,0],dataset_train.iloc[:,1],test_size=0.20,random_state=0)

vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

print(X_train_counts.shape)
print(X_test_counts.shape)

#Chunking to avoid dead kernels(Main memory overflow)
for i in range(1,21):
    df = pd.DataFrame(X_test_counts[(X_test_counts.shape[0]/21)*(i-1)].toarray()[0])
    for j in range((X_test_counts.shape[0]/21)*(i-1)+1,(X_test_counts.shape[0]/21)*(i)-1):
        s = pd.Series(X_train_counts[j].toarray()[0])
        df.merge(pd.DataFrame(s).T)
    df.to_csv('./dataset/processed/processed_x_train' + i + '.csv',sep=',')
    


















