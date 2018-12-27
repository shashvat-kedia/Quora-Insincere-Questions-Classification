import numps as np
import pandas as pd
import matploblib.pyplot as plt
import tensorflow as tf
from preprocess import preprocess, get_preprocessed_batch_data, 

class Bidirectional_LSTM():
    def __init__(self,num_classes,vocab_size,hidden_size,no_of_attention_heads,max_length,batch_size,d_model,d_k):
        self.x = tf.placeholder(shape=[None,None],name='X')
        self.y = tf.placeholder(shape=[None],name='y')
        self.sequence_lengths = tf.placeholder(shape=[None],name='sequence_lengths')
        with tf.variable_scope('encoder_self_attention_head'):
            self.attention = Attention(self.x,no_of_attention_heads,batch_size,max_length,d_model,d_k)
            self.attention_output = self.attention.outputs
            self.attention_output = add_and_norm(self.x,self.attention_output)
            self.attention_output = add_and_norm(self.attention_output,position_wise_feed_forward(self.attention_output))
        cell_fw = tf.contrib.rnn.LSTMCell(hidden_size,forget_bias=1.0,state_is_tuple=True,reuse=tf.get_variable_scope().reuse)
        cell_bw = tf.contrib.rnn.LSTMCell(hidden_size,forget_bias=1.0,state_is_tuple=True,reuse=tf.get_variable_scope().reuse)
        with tf.variable_scope('bidirectional_lstm'):
            (output_fw,output_bw),(output_state_fw,output_state_bw) = tf.nn.bidirectional_dynamic_lstm(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=self.attention_output,
                sequence_length=self.sequence_lengths,
                dtype=tf.float32)
        self.final_state = tf.concat([output_state_fw,output_state_bw],axis=1)
        print(tf.shape(final_output))
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
    
    def add_and_norm(x,trans_x):
        with tf.variable_scope('add_and_norm'):
            return tf.contrib.layers.layer_norm(x + trans_x)
        
    def position_wise_feed_forward(x):
        output_dim = x.get_shape()[-1]
        with tf.variable_scope('position_wise_feed_forward'):
            x = tf.layers.dense(x,2048,activation=tf.nn.relu)
            x = tf.layers.dense(x,output_dim)
            return x
    
class Attention():
    def __init__(self,inputs,no_of_attention_heads,batch_size,pad_length,d_model,d_k):
        self.output = []
        self.inputs = self.positional_encodings(inputs,pad_length,d_model)
        for i in range(0,no_of_attention_heads):
            self.output.append(self.self_attention(self.inputs))
        self.outputs.concat(outputs,axis=2)
        self.outputs = tf.layers.dense(outputs,d_model)
        
    def self_attention(inputs,i,batch_size,pad_length,d_k):
        with tf.variable_scope('self_attention_head'):
            K = tf.layers.dense(inputs,d_k,name='K',activation=tf.nn.relu)
            Q = tf.layers.dense(inputs,d_k,name='Q',activation=tf.nn.relu)
            V = tf.layers.dense(inputs,d_v,name='V',activation=tf.nn.relu)
        mask = tf.ones([pad_length,pad_length])
        mask = tf.reshape(tf.tile(mask,[batch_size,1]),[batch_size,pad_length,pad_length])
        self_attention = tf.matmul(tf.nn.softmax(mask * (tf.matmul(Q,tf.transpose(K,[0,2,1])))/tf.sqrt(tf.to_float(d_k))),V)
        return self_attention
        
    def positional_encodings(inputs,pad_length,d_model):
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
    with open('preprocessed/max_length.txt','r') as file:
        max_length = int(file.read())
    vocab_processor = tf.contrib.learn.preprocessiing.VocabularProcessor(max_length,min_frequency=0).restore('processed/vocab')
    
    with tf.Graph().as_default():
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer)
    
train()