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
import datetime
import tensorflow as tf

def load_and_preprocess(min_frequency=0,vocab_precessor=None):
    starttime = time.time()
    print(os.listdir())
    dataset_train = pd.read_csv('dataset/train.csv')
    dataset_test = pd.read_csv('dataset/test.csv')    
    print(os.listdir('dataset'))
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
    dataset_train.to_csv('dataset/processed_train.csv',sep=',')
    dataset_test.to_csv('dataset/processed_test.csv',sep=',')
    lengths = np.array(list(map(len, [sent.strip().split(' ') for sent in dataset_train.iloc[:0].values])))
    max_length = max(lengths)
    if vocab_processor is None:
        vocab_processor = tf.contrib.preprocessing.VocabularyProcessor(max_length,min_frequency=min_frequency)
        data = np.array(list(vocab_processor.fit_transform(dataset_train.iloc[:,0])))
    else:
        data = np.array(list(vocab_processor.transform(dataset_train.iloc[:,0])))
    data_size = len(data)
    shuffle_index = np.random.permutation(np.arrange(data_size)) 
    dataset_train = dataset_train[shuffle_index]
    lengths = lengths[shuffle_index]
    endtime = time.time()
    print("Time to load and preprocess")
    print(endtime - starttime)
    return dataset_train.iloc[:,0].values,dataset_train.iloc[:,1].values,lengths,vocab_processor
   
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
    
    def __init__(self,num_classes,vocab_size,hidden_size,num_layers,l2_reg_lambda):
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

def train():
    X,y,lengths,vocab_processor = load_and_preprocess(min_frequency=0)
    vocab_processor.save('/processed/vocab')
    X_train,X_test,y_train,y_test,train_lengths,valid_lengths = train_test_split(X,y,lengths,test_size=0.2,random_state=0)
    train_data = get_batch(X_train,y_train,train_lengths,32,50)
    with tf.Graph().as_default():
        with tf.Session() as sess:
            classifier = LSTM(2,len(vocab_processor.vocabulary_._mapping),300,2,0.001)
            global_step = tf.Variable(0,name='global_step',trainable=False)
            learning_rate = tf.train.exponential_decay(1e-3,global_step,100000,1,staircase=True)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            grad_and_vars = optimizer.compute_gradients(classifier.cost)
            train_op = optimizer.apply_gradients(grad_and_vars,global_step=global_step)
            sess.run(tf.global_variables_intializer())
            for train_input in train_data:
                run(train_input,is_training=True)
                current_step = tf.train.global_step(sess,global_step)
            def run(train_input,is_training=True):
                x_data,y_data,length_data = train_input
                fetches = {'step': global_step,
                       'cost': classifier.cost,
                       'accuracy': classifier.accuracy,
                       'learning_rate': learning_rate,
                       'final_state': classifier.final_state}
                feed_dict = {classifier.x: input_x,
                         classifier.y: input_y,
                         classifier.sequence_length: length_data,
                         classifier.batch_size: len(x_data)}
                vars = sess.run(fetches, feed_dict)
                step = vars['step']
                cost = vars['cost']
                accuracy = vars['accuracy']
                if is_training:
                    fetches['train_op'] = train_op
                    fetches['summaries'] = train_summary_op
                    feed_dict[classifier.keep_prob] = 0.5
                else:
                    feed_dict[classifier.keep_prob] = 1.0
                time_str = datetime.datetime.now().isoformat()
                print("{}: step: {}, loss: {:g}, accuracy: {:g}".format(time_str, step, cost, accuracy))
                return accuracy
            
train() 