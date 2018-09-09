# -*- coding:utf-8-*-
import math
import librosa
import numpy as np
import tensorflow as tf
tf.reset_default_graph()
from voiceModel import Model
from spectrumGraph import SpectrumFrames as SF



sample_rate = 10000
max_frames = 3*sample_rate
max_width  = 150
max_height = 80
padding_value = -30


def preprocess(fd_or_array,sr = 10000):
    frames = fd_or_array
    if isinstance(fd_or_array ,str):
        frames,sr = librosa.load(fd_or_array,sr=10000)
    num_slices = math.ceil(frames.shape[0]/max_frames)
    specs = []
    for i in range(num_slices):
        frame_slice = frames[i*max_frames:(i+1)*max_frames]
        frame_spec = np.ones((max_width,max_height))*padding_value
        spec = SF(frame_slice,sr)
        w_,h_ = spec.shape
        w = min(w_,max_width)
        h = min(h_,max_height)
        frame_spec[:w,:h] = spec[:w,:h]
        specs.append(np.expand_dims(frame_spec,axis=0))
    return np.concatenate(specs,axis=0)

def loadLabels(fd):
    labels = {}
    for line in open(fd+'labels.txt','r',encoding='utf8'):
        emo,idx = line.strip().split(' ')
        labels[int(idx)]=emo
    return labels
        

class Predictor():
    def __init__(self,fd='./'):
        self.labels = loadLabels(fd)
        self.model = Model()
        self.sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(self.sess,fd+'model/ser')
    def predict(self,fn_or_array):
        specs = preprocess(fn_or_array)
        prd,logits = self.sess.run([self.model.prediction,
                                   self.model.logits],
                                  feed_dict={self.model.inputs:specs,
                                             self.model.keep_prob:1.0,
                                             self.model.is_training:False})
        rets = [self.labels[prd[i]] for i in range(prd.shape[-1])]
        sum_logits = logits.sum(axis=0)
        logs = [(self.labels[i],sum_logits[i]) for i in range(sum_logits.shape[-1])]
        return rets, logs
            
        
if __name__ == '__main__':
    predictor =  Predictor()
    l,p = predictor.predict('123.m4a')     