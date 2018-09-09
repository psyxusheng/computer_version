# -*- coding: utf-8 -*-

import tensorflow as tf

class Model():
    def __init__(self,
                 width=150,
                 height=80,
                 num_class = 6,
                 ):
        
        def conv2d(inputs,out_filters,kernel,strides,training,vname):
            bn = tf.layers.batch_normalization(inputs,
                                               training=training,
                                               name = vname+'/bn')
            
            acted = tf.nn.relu(bn,name=vname+'/relu')
            
            conved = tf.layers.conv2d(inputs=acted, filters = out_filters,
                                      kernel_size=kernel,
                                      strides=strides,
                                      padding='same',
                                      kernel_initializer = tf.truncated_normal_initializer(stddev=0.02),
                                      activation = None,
                                      name = vname+'/conv2d',
                                      )
            return conved
        
        def max_pool(inputs,kernel,strides,vname):
            return tf.layers.max_pooling2d(inputs,pool_size=kernel,strides=strides,
                                           padding='valid',name = vname+'/max_pooling')
        
        self.inputs = tf.placeholder(name='input',
                                     shape=[None,width,height],
                                     dtype=tf.float32)
        self.targets = tf.placeholder(name='targets',
                                      shape=[None],
                                      dtype=tf.int32)
        self.keep_prob = tf.placeholder(name='keep_prob',shape=[],dtype=tf.float32)
        
        self.is_training = tf.placeholder(name='is_training',shape=[],dtype=tf.bool)
        
        inputs = tf.expand_dims(self.inputs,-1)
        
        with tf.variable_scope('block_1'):
            conv1 = conv2d(inputs,32,(3,3),(1,1),training=self.is_training,vname='conv1')
            conv2 = conv2d(conv1,32,(3,3),(1,1),training=self.is_training, vname='conv2')
            pool3 = max_pool(conv2,(2,2),(2,2),vname='pool3')
        with tf.variable_scope('block_2'):
            conv4 = conv2d(pool3,64,(3,3),(1,1),training=self.is_training,vname = 'conv4')
            conv5 = conv2d(conv4,64,(3,3),(1,1),training=self.is_training,vname = 'conv5')
            pool6 = max_pool(conv5,(2,2),(2,2),vname='pool6')
        with tf.variable_scope('block_3'):
            conv7 = conv2d(pool6,128,(3,3),(1,1),training=self.is_training,vname='conv7')
        with tf.variable_scope('global_pool'):
            global_pool = tf.reduce_mean(conv7,[1,2],name='global_pool')
        with tf.variable_scope('fc'):
            logits_ = tf.layers.dense(global_pool,num_class,name='fc')
            self.logits = tf.nn.dropout(logits_,keep_prob=self.keep_prob)
            
        self.cost = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits = self.logits,
                        labels = self.targets))
        self.prediction = tf.argmax(self.logits,axis=-1)
        correction = tf.equal(self.prediction,tf.cast(self.targets,tf.int64))
        self.accuracy = tf.reduce_mean(tf.cast(correction,tf.float32))

        global_step = tf.Variable(0,name='gs',trainable=False) 
        self.gs = global_step
        learning_rate = tf.train.exponential_decay(
                learning_rate= 1e-2,
                global_step = global_step,
                decay_steps = 1000,
                decay_rate  = 0.90,
                staircase=True,
                )        
        self.learning_rate = learning_rate
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)       
        with tf.variable_scope('trainops'):                      
            with tf.control_dependencies(update_ops):               
                self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.cost,global_step)
