# -*- coding: utf-8 -*-
import numpy as np
from dmodel.denseNet import Model
import tensorflow as tf
import cv2


class Predictor():
    def __init__(self,ModelDir = './dmodel/cpt'):
        self.labels = {int(cls):emo for cls,emo in \
          [line.strip().split(' ') for line in open('labels.txt','r',encoding='utf8')]}
        tf.reset_default_graph()
        self.model = Model(dense_layers=6,num_filters = 32) 
        self.session = tf.Session()
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(self.session,ModelDir)
    def predict(self,img):
        """
        input shape must be [44,44,1]
        """
        width,height,channels = img.shape
        if width!=height or  width <44:
            raise Exception('wrong input dimension')
        if channels !=1:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        if width !=44:
            img = cv2.resize(img,(44,44),cv2.INTER_AREA)
        inputs = np.expand_dims(img,0)
        inputs = np.expand_dims(inputs,-1)
        probs_ = self.session.run(self.model.probalilty,feed_dict={
                                    self.model.images:inputs,
                                    self.model.is_training:False,
                                    self.model.keep_prob:1.0
                                    })
        probs = probs_[0,:]
        label = self.labels[np.argmax(probs)]
        dist = [[self.labels[i],probs[i]] for i in range(probs.shape[0])]
        return label,dist
        
        
        
        
if __name__=='__main__':
    prd = Predictor()