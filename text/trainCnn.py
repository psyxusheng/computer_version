# -*- coding: utf-8 -*-

from textCnn import Model
from dataProcess import Vocab,TextDataSet
import tensorflow as tf
import time

niters = 10000


vocab = Vocab('vocab.json')
vocab_size = vocab.size
model4train = Model(vocab_size)
model4prd = Model(vocab_size,batch_size=1)

trainData = TextDataSet('train.txt',vocab)

session = tf.Session()
session.run(tf.global_variables_initializer())

start = time.time()
for i in range(1,niters+1):
    x,y = trainData.next_batch(64)
    session.run(model4train.trainop,feed_dict={
            model4train.inputs:x,
            model4train.targets:y,
            model4train.keep_prob:0.6})
    if i%1000==0:
        loss,accuracy = session.run([model4train.loss,model4train.accuracy],
                    feed_dict={
                            model4train.inputs:x,
                            model4train.targets:y,
                            model4train.keep_prob:0.6
                            })
        end = time.time()
        print('%10.3f -- %10d -- %10.4f -- %3.2f'%(end-start,i,loss,accuracy))
        start = end

import os
model_dir = 'model'
if not os.path.exists(model_dir):
    os.mkdir(model_dir)  
    
    
saver = tf.train.Saver()    
saver.save(session,model_dir+'/.ckpt')

import numpy as np

from dataProcess import TextDataSet


testData =    TextDataSet('test.txt',vocab) 

X = testData.X
Y = testData.Y

t,r = 0,0

for i in range(X.shape[0]):
    t+=1
    inp = X[i:i+1,:]
    tgt = Y[i]
    prd = session.run([model4prd.predicts],feed_dict={
            model4prd.inputs :inp,
            model4prd.keep_prob:1.0
            })
    p = np.argmax(prd)
    if p==tgt:
        r+=1
print(r/t)  
