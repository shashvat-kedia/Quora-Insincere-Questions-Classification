import numps as np
import pandas as pd
import matploblib.pyplot as plt
import tensorflow as tf
from preprocess import preprocess, get_preprocessed_batch_data, 

class Bidirectional_LSTM():
    def __init__(self,num_classes):
        
    
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