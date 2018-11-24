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

def load_and_preprocess(min_frequency=0,vocab_precessor=None):
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
   max_length = 0
   for idx,row in dataset_train.iterrows():
       length = len(row['question_text'].split(' '))
       if length > max_length:
           max_length = length
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

class LSTM():
    
    def __init__(self,num_classes,vocab_size,):
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.l2_reg_lambda = l2_reg_lambda  #Coefficent term for Regularization using L2 norm
        self.batch_size = tf.placeholder(dtype=tf.int32,shape=[],name="batch_size")
        self.x = tf.placeholder(dtype=tf.float32,shape=[None,None],name='X')
        self.y = tf.placeholder(dtype=tf.float32,shape=[None],name='y')
        self.keep_prob = tf.placeholder(dtype=tf.float32,shape=[],name='dropout_keep_prob')
        self.sequence_length = tf.placeholder(dtype=tf.int32,shape=[None],name='sequence_length')
        self.l2_loss = tf.constant(0.0)
        with tf.device('/cpu:0'), tf.name_scope('embeddiing'):
            embedding = tf.get_variable('embedding',shape=[self.vocab_size,self.hidden_size],dtype=tf.float32)
            inputs = tf.nn.embedding_lookup(embedding,self.x)
        self.inputs = tf.nn.dropout(self.inputs,keep_prob=self.keep_prob)
        cell = tf.contrib.rnn.LSTMCell(self.hidden_size,forget_bias=1.0,state_is_tuple=True,reuse=tf.get_variable_scope().reuse)
        cell = tf.contrib.rnn.DropoutWrapper(cell,output_keep_prob=self.keep_prob)
        cell = tf.contrib.rnn.MultiCellRNN([cell]  * self.num_layers,state_is_tuple=True)
        self.initial_state = cell.zero_state(self.batch_size,dtype=tf.float32)
        with tf.variable_scope('LSTM'):
            outputs,state = tf.nn.dynamic_rnn(cell,inputs=self.inputs,initial_state=self.initial_state,sequence_legnth=self.sequence_length)
        self.final_state = state
        with tf.name_scope('softmax'):
            softmax_w = tf.get_variable('softmax_w',shape=[self.hidden_size,slef.num_classes],dtype=tf.float32)
            softmax_b = tf.get_variable('softmax_b',shape=[self.num_classes],dtype=tf.float32)
        self.l2_loss += tf.nn.l2_loss(softmax_w)
        self.l2_loss += tf.nn.l2_loss(softmax_b)
        self.logits = tf.matmul(self.final_state[self.num_layers-1].h,softmax_w) + softmax_b
        predictions = tf.nn.softmax(self.logits)
        self.prediction = tf.argmax(predictions,1,name='predictions')
        with tf.name_scope('loss'):
            trainable_vars = tf.trainable_vars()
            for var in trainable_vars:
                if 'kernel'  in var.name:
                    self.l2_loss += tf.nn.l2_loss(var)
            losses = tf.nn.sparse_cross_entropy_with_logits(labels=self.y,logits=self.logits)
            self.cost = tf.reduce_mean(losses) + self.l2_reg_lambda * self.l2_loss #Additional loss term added to loss funciton for regularization using L2 norm
        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions,self.y)
            self.correct_num = tf.reduce_sum(tf.cast(correct_predictions, tf.float32))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')
            
    
            
    

