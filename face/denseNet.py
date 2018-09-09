import tensorflow as tf

class Model():
    def __init__(self,
                 width =  44,
                 height = 44,
                 channels = 1,
                 num_cls = 6,
                 first_scale = (3,3),
                 dense_layers = 12,
                 num_filters = 64):
        
        kernel_initializer = tf.truncated_normal_initializer(stddev=0.02)
        
        self.is_training = tf.placeholder(dtype = tf.bool,shape=[],
                                          name='training_flag')
        self.keep_prob = tf.placeholder(dtype = tf.float32,shape=[],
                                        name='keep_prob')
        
        def conv2d(inputs,out_filters,kernel,strides,vname):
            with tf.variable_scope(vname):
                bn = tf.layers.batch_normalization(inputs,
                                                   training=self.is_training,
                                                   name =vname+'/bn' )
                acted = tf.nn.relu(bn,name =vname+'/relu')
                conved = tf.layers.conv2d(inputs = acted,
                                          filters = out_filters,
                                          kernel_size=kernel,
                                          strides=strides,
                                          name =vname+'/conved',
                                          padding='same',
                                          kernel_regularizer= None,
                                          kernel_initializer=kernel_initializer)
                return conved

        def conv_concate(x,growth_rate,name,layer_num):
            with tf.variable_scope(name):
                l=conv2d(x,out_filters = growth_rate,kernel = (3,3),strides = (1,1),vname='/conv%d'%layer_num)
                l=tf.concat([l,x],3,name = 'concat%d'%layer_num)
            return l

        def dense_block(l,layers=12,growth_rate=12):
            for i in range(layers):
                l = conv_concate(l,growth_rate = growth_rate, name = 'dense_blcok_{}'.format(i+1),layer_num=i+1)
            return l

        def transition(l,name=None):
            l=conv2d(l,out_filters=16,kernel=(1,1),strides = (1,1),vname = name+'/conv_1by1')
            l=tf.layers.average_pooling2d(inputs=l,pool_size=(2,2),strides=(2,2),padding='valid')
            return l

        self.images = tf.placeholder(shape=[None,width,height,channels],dtype=tf.float32,
                                     name='images')
        
        
        
        self.labels = tf.placeholder(shape=[None],dtype=tf.int32,name='labels')


        images = tf.div(self.images,255.0)
        
        first_layer = tf.layers.conv2d(inputs = images,filters=num_filters,kernel_size=first_scale,
                                       strides = (1,1),name = 'conv0',
                                       padding='same',kernel_initializer = kernel_initializer)
    
        # dense block 1
        with tf.variable_scope('block1'):
            l=dense_block(first_layer,layers = dense_layers,growth_rate=num_filters)
            l=transition(l,name = 'transition')
        with tf.variable_scope('block2'):
            l=dense_block(l,layers = dense_layers,growth_rate=num_filters)
            l=transition(l,name = 'transition')
    
        with tf.variable_scope('block3'):
            l=dense_block(l,layers = dense_layers,growth_rate=num_filters)
        l=tf.layers.batch_normalization(l,training=self.is_training,name='bn')
        l=tf.nn.relu(l,name='relu')
        
        # global avgpool
        pooled=tf.reduce_mean(l,[1,2],name='gloabel_pooling')
        
        logits_ = tf.layers.dense(pooled,num_cls,name='fc')
        
        logits = tf.nn.dropout(logits_,keep_prob=self.keep_prob)
        
        
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels,
                                                                   logits=logits))
        
        
        self.predicts = logits
        
        self.prediction = tf.argmax(logits,axis=-1)
        
        self.probalilty  = tf.nn.softmax(logits,axis=-1)
        
        correct_prediction = tf.equal(self.prediction, tf.to_int64(self.labels))
        
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
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
        
        with tf.variable_scope('trainops'):
            
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            
            with tf.variable_scope('train_op', reuse = tf.AUTO_REUSE):
            
                with tf.control_dependencies(update_ops):
                
                    self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss,global_step)
                    
if __name__=='__main__':
    model = Model()


