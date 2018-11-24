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

def load_and_preprocess(min_frequency=0,max_length=0,vocab_precessor=None):
    starttime = time.time()
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
    #stemmer = SnowballStemmer('english')
    #def lemmatize(text):
        #return stemmer.stem(WordNetLemmatizer().lemmatize(text))
    for idx,row in dataset_train.iterrows():
        nval = ''
        for val in row['question_text'].split(' '):
            val = re.sub('[^A-Za-z]+',' ',val)
            if(val != ' '):
                if val.lower() not in stop:
                    nval = nval + ' ' + val.lower()
        dataset_train.at[idx,'question_text'] = nval
    for idx,row in dataset_test.iterrows():
        nval = ''
        for val in row['question_text'].split(' '):
            val = re.sub('[^A-Za-z]+',' ',val)
            if(val != ' '):
                if val.lower() not in stop:
                    nval = nval + ' ' + val.lower()
        dataset_test.at[idx,'question_text'] = nval        
   print(dataset_train.head())
   print(dataset_test.head())
   dataset_train.to_csv('./dataset/processed_train.csv',sep=',')
   dataset_test.to_csv('./dataset/processed_test.csv',sep=',')
   if vocab_processor is None:
       vocab_processor = tf.contrib.preprocessing.VocabularyProcessor(max_length,min_frequency=min_frequency)
       data = np.array(list(vocab_processor.fit_transform(dataset_train.iloc[:,0])))
   else:
       data = np.array(list(vocab_processor.transform(dataset_train.iloc[:,0])))
   data_size = len(data)
   shuffle_index = np.random.permutation(np.arrange(data_size)) 
   shuffled_dataset_train = dataset_train[shuffle_index]
    X_train,X_test,y_train,y_test = train_test_split(shuffled_dataset_train.iloc[:,0],shuffled_dataset_train.iloc[:,1],test_size=0.20,random_state=0)
   endtime = time.time()
   print("Time to load and preprocess")
   print(endtime - starttime)
   return X_train,X_test,y_train,y_test
   
def get_batch(data,labels,lengths,batch_size,epochs):
    assert len(data) == len(labels) == len(lengths)
    no_batches = len(data) // batch_size
    for i in range(1,epochs):
        for j in range(1,no_batches):
            start_index = j * batch_size
            end_index = start_index + batch_size
            x_data = data[start_index:end_index]
            y_data = labels[start_index:end_index]
            length_data = lengths[start_index:end_index]
            yield x_data,y_data,length_data


            
    

