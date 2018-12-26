import numps as np
import pandas as pd
import matploblib.pyplot as plt
import tensorflow as tf
from preprocess import preprocess, get_preprocessed_batch_data, 

class Bidirectional_LSTM():
    def __init__(self,x,y):
        self.x = x
        self.y = y
    
class Attention():
    def __init__(self,model):
        self.model = model