import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime
from preprocess import preprocess, get_processed_batch_data, save_testing_data

class LSTM():
    
    def __init__(self,num_classes,vocab_size,hidden_size,num_layers,l2_reg_lambda):
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.l2_reg_lambda = l2_reg_lambda  #Coefficent term for Regularization using L2 norm
        self.batch_size = tf.placeholder(dtype=tf.int32,shape=[],name="batch_size")
        self.x = tf.placeholder(dtype=tf.int32,shape=[None,None],name='X')
        self.y = tf.placeholder(dtype=tf.int64,shape=[None],name='y')
        self.keep_prob = tf.placeholder(dtype=tf.float32,shape=[],name='dropout_keep_prob')
        self.sequence_length = tf.placeholder(dtype=tf.int32,shape=[None],name='sequence_length')
        self.l2_loss = tf.constant(0.0)
        with tf.device('/cpu:0'), tf.name_scope('embeddiing'):
            embedding = tf.get_variable('embedding',shape=[self.vocab_size,self.hidden_size],dtype=tf.float32)
            self.inputs = tf.nn.embedding_lookup(embedding,self.x)
        self.inputs = tf.nn.dropout(self.inputs,keep_prob=self.keep_prob)
        cell = tf.contrib.rnn.LSTMCell(self.hidden_size,forget_bias=1.0,state_is_tuple=True,reuse=tf.get_variable_scope().reuse)
        cell = tf.contrib.rnn.DropoutWrapper(cell,output_keep_prob=self.keep_prob)
        cell = tf.nn.rnn_cell.MultiRNNCell([cell]  * self.num_layers,state_is_tuple=True)
        self.initial_state = cell.zero_state(self.batch_size,dtype=tf.float32)
        with tf.variable_scope('LSTM'):
            outputs,state = tf.nn.dynamic_rnn(cell,inputs=self.inputs,initial_state=self.initial_state,sequence_length=self.sequence_length)
        self.final_state = state
        with tf.name_scope('softmax'):
            softmax_w = tf.get_variable('softmax_w',shape=[self.hidden_size,self.num_classes],dtype=tf.float32)
            softmax_b = tf.get_variable('softmax_b',shape=[self.num_classes],dtype=tf.float32)
        self.l2_loss += tf.nn.l2_loss(softmax_w)
        self.l2_loss += tf.nn.l2_loss(softmax_b)
        self.logits = tf.matmul(self.final_state[self.num_layers-1].h,softmax_w) + softmax_b
        predictions = tf.nn.softmax(self.logits)
        self.predictions = tf.argmax(predictions,1,name='predictions')
        with tf.name_scope('loss'):
            trainable_vars = tf.trainable_variables()
            for var in trainable_vars:
                if 'kernel'  in var.name:
                    self.l2_loss += tf.nn.l2_loss(var)
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y,logits=self.logits)
            #Additional loss term added to loss funciton for regularization using L2 norm
            self.cost = tf.reduce_mean(losses) + self.l2_reg_lambda * self.l2_loss 
        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions,self.y)
            self.correct_num = tf.reduce_sum(tf.cast(correct_predictions, tf.float32))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')
            
def train():
    preprocess()
    max_length = 0
    with open('processed/max_length.txt','r') as file:
        max_length = int(file.read())
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_length,min_frequency=0).restore('processed/vocab')
    chunksize = 10 ** 4
    count = 1
    for epoch in range(1,50):
        for data in pd.read_csv('dataset/processed_train.csv',chunksize=chunksize):
            print('Batch no.:-')
            print(count)
            count += 1
            print(data.info(memory_usage='deep'))
            data = data.drop(data.columns[0],axis=1)
            train_data = get_processed_batch_data(data,vocab_processor,32)
            with tf.Graph().as_default():
                with tf.Session() as sess:
                    classifier = LSTM(2,len(vocab_processor.vocabulary_._mapping),300,2,0.001)
                    global_step = tf.Variable(0,name='global_step',trainable=False)
                    learning_rate = tf.train.exponential_decay(1e-3,global_step,100000,1,staircase=True)
                    optimizer = tf.train.AdamOptimizer(learning_rate)
                    grad_and_vars = optimizer.compute_gradients(classifier.cost)
                    train_op = optimizer.apply_gradients(grad_and_vars,global_step=global_step)
                    sess.run(tf.global_variables_initializer())
                    def run(train_input,is_training=True):
                        x_data,y_data,length_data = train_input
                        fetches = {'step': global_step,
                                   'cost': classifier.cost,
                                   'accuracy': classifier.accuracy,
                                   'learning_rate': learning_rate,
                                   'final_state': classifier.final_state}
                        feed_dict = {classifier.x: x_data,
                                     classifier.y: y_data,
                                     classifier.sequence_length: length_data,
                                     classifier.batch_size: len(x_data)}
                        vars = sess.run(fetches, feed_dict)
                        step = vars['step']
                        cost = vars['cost']
                        accuracy = vars['accuracy']
                        if is_training:
                            fetches['train_op'] = train_op
                            feed_dict[classifier.keep_prob] = 0.5
                        else:
                            feed_dict[classifier.keep_prob] = 1.0
                            time_str = datetime.datetime.now().isoformat()
                            print("{}: step: {}, loss: {:g}, accuracy: {:g}".format(time_str, step, cost, accuracy))
                        return accuracy
                    for train_input in train_data:
                        run(train_input,is_training=True)
        current_step = tf.train.global_step(sess,global_step)
        print("Current step:- ")
        print(current_step)
            
train() 