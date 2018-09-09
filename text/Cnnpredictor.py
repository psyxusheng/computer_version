# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from textCnn import Model
from dataProcess import Vocab
import  HyperParameter as HP
from funcs import timeit
time_steps = HP.time_steps

from tqdm import trange

model = Model(vocab_size=3501)


session = tf.Session()
saver = tf.train.Saver()
saver.restore(session,'model/.ckpt')

def loadLabels():
    labels = {}
    for line in open('labels.txt','r',encoding='utf8'):
        _,_,name,index = line.strip().split('\t')
        labels[int(index)]=name
    return labels

labels = loadLabels()

def softmax(array):
    array = array-array.mean()
    exps = np.exp(array)
    return exps / exps.sum()


class Predictor():
    def __init__(self,p=0.4):
        self._p = p
        self._vocab = Vocab('vocab.json')
        self._labels = loadLabels()
    #@timeit
    def predict(self,sentence , type_ ='str'):
        if type_ =='str':
            ids = self._vocab.sentence2id(sentence)
            ids = ids[:time_steps]
            ids = ids+[0]*(time_steps-len(ids))
            inputs = [ids]
        else:
            inputs = sentence
        prd = session.run(model.predicts,feed_dict={
                model.inputs:inputs,
                model.keep_prob:1.0,
                })
        probs = softmax(prd[0,:])
        #maxval = probs.max()
        maxpos = np.argmax(probs)
        return self._labels[maxpos],probs,prd[0,:]
        
if __name__=='__main__':
    predictor= Predictor()
    p,r,l = predictor.predict('开心')
