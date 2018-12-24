#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 14:20:53 2018

@author: shashvatkedia
"""

class Bidirectional_LSTM():
    def __init__(self,x,y):
        self.x = x
        self.y = y
    
class Attention():
    def __init__(self,model):
        self.model = model