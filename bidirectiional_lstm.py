import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime
from preprocess import preprocess, get_transformed_batch_data 

class Bidirectional_LSTM():
    def __init__(self,num_classes,hidden_size,no_of_attention_heads,max_length,batch_size,d_model,d_k,d_v,stack_size):
        self.x = tf.placeholder(shape=[None,max_length,hidden_size],dtype=tf.float32,name='X')
        self.y = tf.placeholder(shape=[None],dtype=tf.float32,name='y')
        self.sequence_lengths = tf.placeholder(shape=[None],dtype=tf.float32,name='sequence_lengths')
        with tf.variable_scope('multi_head_self_attention'):
            self.attention = Attention(self.x,no_of_attention_heads,batch_size,max_length,d_model,d_k,d_v,stack_size)
        self.attention_output = self.attention.output       
        cell_fw = tf.contrib.rnn.LSTMCell(hidden_size,forget_bias=1.0,state_is_tuple=True,reuse=tf.get_variable_scope().reuse)
        cell_bw = tf.contrib.rnn.LSTMCell(hidden_size,forget_bias=1.0,state_is_tuple=True,reuse=tf.get_variable_scope().reuse)
        with tf.variable_scope('bidirectional_lstm'):
            (output_fw,output_bw),(output_state_fw,output_state_bw) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=self.attention_output,
                sequence_length=self.sequence_lengths,
                dtype=tf.float32)
        self.final_state = tf.concat([output_state_fw,output_state_bw],axis=1)
        print(tf.shape(self.final_state))
        with tf.variable_scope('softmax'):
            self.softmax_w = tf.get_variable('softmax_w',shape=[2*hidden_size,num_classes],initializer=tf.truncated_normal_initializer(),dtype=tf.float32)
            self.softmax_b = tf.get_variable('softmax_b',shape=[num_classes],initializer=tf.constant_initializer(0.0),dtype=tf.float32)
        self.logits = tf.matmul(self.final_state,self.softmax_w) + self.softmax_b
        predictions = tf.nn.softmax(self.logits)
        self.predictions = tf.argmax(predictions,1,name='predictions')
        with tf.name_socpe('loss'):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(self.y,self.logits)
            self.cost = tf.reduce_mean(losses)
        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions,self.y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions,tf.float32),name='accuracy')
    
class Attention():
    def __init__(self,inputs,no_of_attention_heads,batch_size,pad_length,d_model,d_k,d_v,stack_size):
        self.output = []
        self.inputs = self.positional_encodings(inputs,pad_length,d_model)
        for i in range(0,stack_size):
            with tf.variable_scope('encoder_block_' + str(i)):
                with tf.variable_scope('encoder',reuse=True):
                    for i in range(0,no_of_attention_heads):
                        self.output.append(self.self_attention(self.inputs,batch_size,pad_length,d_k,d_v))
                self.output = tf.concat(self.output,axis=2)
                self.output = tf.layers.dense(self.output,d_model)
                self.output = self.add_and_norm(self.inputs,self.output)
                self.output = self.add_and_norm(self.output,self.position_wise_feed_forward(self.output))
                self.inputs = self.output
                if i != stack_size - 1:
                    self.output = []
        
    def self_attention(self,inputs,batch_size,pad_length,d_k,d_v):
        with tf.variable_scope('self_attention_head',reuse=tf.AUTO_REUSE):
            K = tf.layers.dense(inputs,d_k,name='K',activation=tf.nn.relu)
            Q = tf.layers.dense(inputs,d_k,name='Q',activation=tf.nn.relu)
            V = tf.layers.dense(inputs,d_v,name='V',activation=tf.nn.relu)
        mask = tf.ones([pad_length,pad_length])
        mask = tf.reshape(tf.tile(mask,[batch_size,1]),[batch_size,pad_length,pad_length])
        self_attention = tf.matmul(tf.nn.softmax(mask * (tf.matmul(Q,tf.transpose(K,[0,2,1])))/tf.sqrt(tf.to_float(d_k))),V)
        return self_attention
    
    def add_and_norm(self,x,trans_x):
        with tf.variable_scope('add_and_norm',reuse=tf.AUTO_REUSE):
            return tf.contrib.layers.layer_norm(x + trans_x)
        
    def position_wise_feed_forward(self,x):
        output_dim = x.get_shape()[-1]
        with tf.variable_scope('position_wise_feed_forward'):
            x = tf.layers.dense(x,2048,activation=tf.nn.relu)
            x = tf.layers.dense(x,output_dim)
            return x
        
    def positional_encodings(self,inputs,pad_length,d_model):
        def sincos(x,i):
            if i%2 == 0:
                return np.sin(x)
            return np.cos(x)
        with tf.variable_scope('positional_encodings'):
            pe = tf.convert_to_tensor([sincos(pos/(10000**(2*i/d_model)),i) for pos in range(1,pad_length+1) for i in range(1,d_model+1)])
            pe = tf.reshape(pe,[-1,pad_length,d_model])
            return tf.add(inputs,pe)
    
def train():
    preprocess()
    max_length = 0
    batch_size = 32
    chunksize = 10016
    epochs = 50
    with open('processed/max_length.txt','r') as file:
        max_length = int(file.read())
    with tf.Graph().as_default():
        with tf.Session() as sess:
            classifier = Bidirectional_LSTM(2,300,8,max_length,batch_size,300,64,64,6)
            global_step = tf.Variable(0,name='global_step',trainable=False)
            learning_rate = tf.train.exponential_decay(1e-3,global_step,15,1,staircase=True)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            grads = optimizer.compute_gradients(classifier.cost)
            train_op = optimizer.apply_gradients(grads,global_step=global_step)
            sess.run(tf.global_variables_initializer())
            def run(train_input,is_training=True):
                x_data,y_data,length_data = train_input
                fetches = {
                        'step': global_step,
                        'cost': classifier.cost,
                        'accuracy': classifier.accuracy,
                        'learning_rate': learning_rate,
                        'final_state': classifier.final_state
                        }
                feed_dict = {
                        classifier.x: x_data,
                        classifier.y: y_data,
                        classifier.sequence_lengths: length_data
                        }
                vars = sess.run(fetches,feed_dict)
                step = vars['step']
                cost = vars['cost']
                accuracy = vars['accuracy']
                if is_training:
                    fetches['train_op'] = train_op
                else:
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step: {}, loss: {:g}, accuracy: {:g}".format(time_str, step, cost, accuracy))
                return accuracy
            for epoch in range(0,epochs):
                for data in pd.read_csv('dataset/processed_train.csv',chunksize=chunksize):
                    print(data.info(memory_usage='deep'))
                    data = data.drop(data.columns[0],axis=1)
                    train_data = get_transformed_batch_data(data,max_length,batch_size,chunksize)
                    for train_input in train_data:
                        run(train_input,is_training=True)
                    current_step = tf.train.global_step(sess,global_step)
                    print('Current step:- ')
                    print(current_step)
    
train()
