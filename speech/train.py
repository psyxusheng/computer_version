# -*- coding: utf-8 -*-
import os,time
import tensorflow as tf
from dataProcess import DataSet
from voiceModel import Model

model = Model()

session = tf.Session()
session.run(tf.global_variables_initializer())

data = DataSet()

start = time.time()

for i in range(1,10001):
    x,y = data.next_batch(100)
    session.run(model.train_op,feed_dict={model.inputs:x,
                                          model.targets:y,
                                          model.keep_prob:0.6,
                                          model.is_training:True,})
    if i%100==0:
        end = time.time()
        loss,acc = session.run([model.cost,model.accuracy],feed_dict={model.inputs:x,
                                                           model.targets:y,
                                                           model.keep_prob:1.0,
                                                           model.is_training:False,})
        
        print(round(end-start,1),i,loss,acc)
        start = end
        
model_dir = 'model'
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
saver = tf.train.Saver(tf.global_variables())
saver.save(session,'ser')


t,r=0.0,0.0
for x,y in data.export():
    prd = session.run(model.prediction,feed_dict={model.inputs:x,
                                            model.targets:y,
                                            model.keep_prob:1.0,
                                            model.is_training:False,})
    t+=1
    if prd[0]==y:
        r+=1
    
print(r/t)