# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np


class Model():
    epsilon = 1e-8
    def __init__(self,
                 batch_size = 32,
                 num_residual_blocks = [2,4,4,2],
                 num_filter_base = 64,
                 class_num = 8,  
                 input_width  = 32,
                 input_height = 32,
                 input_channels = 3,
                 is_trainning = True,
                 first_scale = (3,3),
                 act_func_name = 'relu',
                 ):
        """
        fine-tune determines which layers to be kept fixed
        """
            
        def normalize(inputs): 
            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
            normalized = (inputs - mean) / ( (variance + self.epsilon) ** (.5) )     
            return normalized
        
        
        
        if act_func_name == 'relu':
            act_func = tf.nn.relu
        elif act_func_name == 'leaky_relu':
            act_func = tf.nn.leaky_relu
        


        def residual_block(x, output_channel , act_func ,is_trainning = True):
            """residual connection implementation"""
            """layer has a parameter 'trainnable' , help to fixed fine-tune"""
            input_channel = x.get_shape().as_list()[-1]
            if input_channel * 2 == output_channel:
                increase_dim = True
                strides = (2, 2)
            elif input_channel == output_channel:
                increase_dim = False
                strides = (1, 1)
            else:
                raise Exception("input channel can't match output channel")
            conv1 = tf.layers.conv2d(x,
                                     output_channel,
                                     (3,3),
                                     strides = strides,
                                     padding = 'same',
                                     kernel_initializer = tf.truncated_normal_initializer(stddev=0.02),
                                     activation = act_func,
                                     name = 'conv1')
            
            conv2 = tf.layers.conv2d(conv1,
                                     output_channel,
                                     (3, 3),
                                     strides = (1, 1),
                                     padding = 'same',
                                     kernel_initializer = tf.truncated_normal_initializer(stddev=0.02),
                                     activation = act_func,
                                     name = 'conv2')
            
            if increase_dim:
                # [None, image_width, image_height, channel] -> [,,,channel*2]
                pooled_x = tf.layers.average_pooling2d(x,
                                                       (2, 2),
                                                       (2, 2),
                                                       padding = 'valid')
                padded_x = tf.pad(pooled_x,
                                  [[0,0],
                                   [0,0],
                                   [0,0],
                                   [input_channel // 2, input_channel // 2]])
            else:
                padded_x = x
            
            output_x = conv2 + padded_x
        
            return output_x
        
        def res_net(x, 
                    num_residual_blocks, 
                    num_filter_base,
                    class_num,
                    is_trainning=True):
            """residual network implementation"""
            """
            Args:
            - x:
            - num_residual_blocks: eg: [3, 4, 6, 3]
            - num_filter_base:
            - class_num:
            """
            num_subsampling = len(num_residual_blocks)
            layers = []
            # x: [None, width, height, channel] -> [width, height, channel]
            input_size = x.get_shape().as_list()[1:]
            with tf.variable_scope('conv0',reuse = tf.AUTO_REUSE):
                conv0 = tf.layers.conv2d(x, 
                                         num_filter_base,
                                         first_scale,
                                         strides = (1, 1),
                                         padding = 'same',
                                         kernel_initializer = tf.truncated_normal_initializer(stddev=0.02),
                                         activation = tf.nn.relu,
                                         name = 'conv0')
                layers.append(conv0)
            # eg:num_subsampling = 4, sample_id = [0,1,2,3]
            for sample_id in range(num_subsampling):
                for i in range(num_residual_blocks[sample_id]):
                    with tf.variable_scope("conv%d_%d" % (sample_id, i),reuse=tf.AUTO_REUSE):
                        conv = residual_block(
                            layers[-1],
                            num_filter_base * (2 ** sample_id),
                            act_func,
                            is_trainning)
            
                        layers.append(conv)
            multiplier = 2 ** (num_subsampling - 1)
            assert layers[-1].get_shape().as_list()[1:] \
                == [input_size[0] / multiplier,
                    input_size[1] / multiplier,
                    num_filter_base * multiplier]
            with tf.variable_scope('fc',reuse=tf.AUTO_REUSE):
                # layer[-1].shape : [None, width, height, channel]
                # kernal_size: image_width, image_height
                global_pool = tf.reduce_mean(layers[-1], [1,2])
                logits = tf.layers.dense(global_pool, class_num)
                if is_trainning :
                    logits = tf.nn.dropout(logits,keep_prob=0.5)
            return layers[-1],logits

        self.x = tf.placeholder(shape=[batch_size,input_width,input_height,input_channels],
                                      dtype = tf.float32)
        
        self.y = tf.placeholder(shape=[batch_size],dtype=tf.int32)
        
        input_images = self.x
        
        if is_trainning:
            ##image_aug
            images = tf.split(self.x, num_or_size_splits=batch_size,axis=0)
        
            aug_images = []
            for image in images:
                image_ = tf.reshape(image,[input_width,input_height,input_channels])
                flipped = tf.image.random_flip_left_right(image_)
                contr   = tf.image.random_contrast(flipped,lower=0.5,upper=1.8)
                aug_images.append(tf.expand_dims(contr,0))
            
            input_images = tf.concat(aug_images,axis=0) 
            
        inpupt_images_scaled = tf.div(input_images,tf.reduce_max(input_images))
        
        
        self.tmp,self.y_ = res_net(inpupt_images_scaled, num_residual_blocks, 
                                num_filter_base, class_num,
                                is_trainning = is_trainning)

        self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.y, 
                                                           logits=self.y_)

        predict = tf.argmax(self.y_, 1)
        
        self.prediction = predict
        self.probablity = tf.nn.softmax(self.y_)

        correct_prediction = tf.equal(predict, tf.to_int64(self.y))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.variable_scope('train_op', reuse = tf.AUTO_REUSE):
            with tf.control_dependencies(update_ops):
                self.train_op = tf.train.AdamOptimizer(1e-3).minimize(self.loss)
            
if __name__=='__main__':
    m = Model()