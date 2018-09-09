# -*- coding: utf-8 -*-
import tensorflow as tf

class Model():
    """
    res-net v2 
    with spatial prymiad pooling which could let the model dealing variable inputs size
    default parameters just for picture with 224*224
    
    """
    def __init__(self,
                 num_cls,
                 init_lr = 1e-3,
                 blocks = [64]*3+[128]*6+[256]*4+[512]*3,
                 input_channels = 1,
                 init_scale = (3,3),
                 init_strides = (1,1),
                 reuse = False,
                 ):
        cnn_init = tf.truncated_normal_initializer(stddev=0.02)
        base_filter = blocks[0]
        def res_block(inputs,out_filters,training,scope):
            """
            res-net v2 : https://arxiv.org/pdf/1603.05027.pdf
            use (1,1)+(3,3)+(1,1) replace for (3,3)+(3,3)
            first do sub-dimension such as using (1,1) reduce channels from 256 -> 64
            then do (3,3) conv keep the channel unchange 64
            then do(1,1) increase the channels (1,1) from 64-->256
            
            """
            inp_filters = inputs.get_shape().as_list()[-1]
            
            if inp_filters % 4!= 0:
                
                raise Exception('number of filters should divsion by 4!!')
                
            
            
            #bn0 = tf.layers.batch_normalization(inputs = inputs,training = training,name = scope+'_bn0',)
            
            #relu0 = tf.nn.leaky_relu(features = bn0, alpha = 0.2,name = scope+'_relu10')
            
            if inp_filters != out_filters:
                strides = (2,2)
                orig_x = tf.layers.conv2d(inputs,filters=out_filters,
                                          kernel_size=(1,1),
                                          padding='same',strides = (2,2),name=scope+'_upsampling',
                                          kernel_initializer=cnn_init)
            else:
                strides = (1,1)
                orig_x = inputs
            
            
            bn1 = tf.layers.batch_normalization(inputs = inputs,
                                                training = training,name = scope+'_bn1',)
            
            relu1 = tf.nn.leaky_relu(features = bn1, alpha = 0.2,name = scope+'_relu1')
            
            conv1 = tf.layers.conv2d(inputs = relu1,filters = inp_filters// 4 , 
                                     kernel_size=(1,1),strides=strides,
                                     kernel_initializer=cnn_init,
                                     activation=None,
                                     padding='same',name=scope+'_conv1')
            
            bn2 = tf.layers.batch_normalization(conv1,training=training,name=scope+'_bn2')
            
            relu2 = tf.nn.leaky_relu(features = bn2, alpha = 0.2,name = scope+'_relu2')
            
            conv2 = tf.layers.conv2d(inputs = relu2 , filters=inp_filters//4,
                         kernel_size=(3,3),strides=(1,1),
                         kernel_initializer=cnn_init,
                         activation=None,
                         padding='same',name=scope+'_conv2')
            
            bn3 = tf.layers.batch_normalization(conv2,training=training,name = scope+'_bn3')
            
            relu3 = tf.nn.leaky_relu(features = bn3, alpha = 0.2,name = scope+'_relu3')
            
            conv3 = tf.layers.conv2d(inputs = relu3 , filters=out_filters,
                         kernel_size=(1,1),strides=(1,1),
                         kernel_initializer=cnn_init,
                         activation=None,
                         padding='valid',name=scope+'_conv3')
            
            output = tf.add(orig_x,conv3,name=scope+'_shortcut')
            
            return output
        
        self.images = tf.placeholder(shape=[None,None,None,input_channels],
                                     name = 'input_images',
                                     dtype=tf.float32)
        self.training = tf.placeholder(shape=[],dtype=tf.bool,name='training_flag')
        
        self.labels = tf.placeholder(shape=[None],dtype=tf.int32,name = 'labels')
        
        self.keep_prob = tf.placeholder(shape=[],dtype=tf.float32,name='keep_prob')
        
        with tf.variable_scope('conv0'):
            
            bn0 = tf.layers.batch_normalization(inputs = self.images,
                                               training=self.training,
                                               name = 'bn0')
            
            relu0 = tf.nn.leaky_relu(bn0,alpha=0.2,name='relu0')
            
            conv0 = tf.layers.conv2d(inputs=relu0,filters = base_filter,
                                     kernel_size = init_scale,
                                     strides = (2,2),padding='same',
                                     kernel_initializer=cnn_init,
                                     activation=None,name = 'conv0')
            
            pool0 = tf.layers.max_pooling2d(inputs = conv0,pool_size=(3,3),
                                            strides = init_strides,
                                            padding ='same', name= 'pool0')
        conv =pool0
        for block_id,output_filters in enumerate(blocks):
            conv = res_block(conv,output_filters,self.training,scope='block_%d'%(block_id+1))
        
        with tf.variable_scope('global_pooling'):
            gl = tf.reduce_mean(conv,[1,2],name='globalpooling')
        
        with tf.vaiable_scope('fc'):
            logits = tf.layers.dense(gl,num_cls,activation=None,
                                     name='fc')
        self.predicts = logits
        self.prediction = tf.argmax(logits,axis=-1)
        self.probablity = tf.nn.softmax(logits,axis=-1)
        self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels,
                                                                                  logits=logits))
        gs = tf.Variable(0,dtype=tf.int32,name='global_step',
                         trainable=False)
        lr = tf.train.exponential_decay(learning_rate=init_lr,
                                        global_step=gs,
                                        decay_step=1000,
                                        decay_rate=0.9,
                                        staircase=True,
                                        name='learning_rate')
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.cost,gs)
            
        
        
if __name__=='__main__':
    model = Model()            
            
        
            