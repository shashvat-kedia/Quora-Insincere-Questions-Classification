import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from nltk import word_tokenize
from nltk.corpus import stopwords
import re
from sklearn.model_selection import train_test_split
from nltk.stem import PorterStemmer
import time
import pickle

embeddings = {}

def preprocess(path):
    starttime = time.time()
    dataset = pd.read_csv(path)
    if('target' in dataset):
        print('Preprocessing train dataset')
        print(dataset.groupby(['target']).size())
        savepath = 'dataset/processed_train.csv'
    else:
        print('Preprocessing test dataset')
        savepath = 'dataset/processed_test.csv'
    stop = set(stopwords.words('english'))
    print(dataset.shape)
    print(dataset.head())
    dataset = dataset.drop(['qid'],axis=1) 
    ps = PorterStemmer()
    for idx,row in dataset.iterrows():
        nval = ''
        for val in row['question_text'].split(' '):
            val = re.sub('[^A-Za-z]+',' ',val)
            if(val != ' '):
                if val.lower() not in stop:
                    nval = nval + ' ' + ps.stem(val.lower())
        dataset.at[idx,'question_text'] = nval
    print(dataset.shape)
    print(dataset.head())
    dataset.to_csv(savepath,sep=',')
    endtime = time.time()
    print("Time for preprocessing")
    print(endtime - starttime)
    
def create_vocabulary(min_frequency=0):
    starttime = time.time()
    dataset = pd.read_csv('dataset/processed_train.csv')
    dataset = dataset.drop(dataset.columns[0],axis=1)
    print(dataset.shape)
    print(dataset.head())
    max_length = 0
    for idx,row in dataset.iterrows():
        length = len(str(row['question_text']).strip().split())
        if length > max_length:
            max_length = length
    print("Max length:- ")
    print(max_length)
    with open('processed/max_length.txt','w') as file:
        file.write(str(max_length))  
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_length,min_frequency=min_frequency)
    for idx,row in dataset.iterrows():
        vocab_processor.fit(str(row['question_text']))
    vocab_processor.save('processed/vocab')
    endtime = time.time()
    print('Time to create vocabulary')
    print(endtime - starttime)

def get_processed_batch_data(data,vocab_processor,batch_size,chunksize):
    row_data = []
    lengths = []
    labels = []
    no_of_batches = chunksize // batch_size
    for idx,row in data.iterrows():
        row_data.append(str(row['question_test']))
        lengths.append(len(str(row['question_text']).strip().split(' ')))
        labels.append(row['target'])
    for i in range(0,no_of_batches):
        start_index = i * batch_size
        end_index = start_index + batch_size
        data_batch = row_data[start_index:end_index]
        y_data = labels[start_index:end_index]
        length_data = lengths[start_index:end_index]
        x_data = []
        for k in range(0,len(data_batch)):
            x_data.append(list(vocab_processor.transform(str(data_batch[i]))))
        x_data= np.array(x_data)
        y_data = np.array(y_data)
        length_data = np.array(length_data)
        yield x_data,y_data,length_data

def embedding_lookup(x):
    if(len(embeddings) == 0):
        populate_embeddings_dict()
    words = x.split()
    embedding = []
    for i in range(0,len(words)):
        if(words[i] in embeddings):
            embedding.append(embeddings[words[i]])
    embedding = np.array(embedding)
    return embedding

def populate_embeddings_dict():
    with open('data/embedding/glove.6B.300d.txt','r') as file:
        for line in file:
            values = line.split()
            word = values[0]
            word_embedding = np.asarray(values[1:])
            embeddings[word] = word_embedding

def preprocess():
    if(not os.path.isfile('dataset/processed_train.csv')):
        preprocess('dataset/train.csv')
    if(not os.path.isfile('dataset/processed_test.csv')):
        preprocess('dataset/test.csv')
    if(not os.path.isfile('processed/vocab')):
        create_vocabulary()
    populate_embeddings_dict()