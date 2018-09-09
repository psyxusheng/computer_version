import tensorflow as tf 

class Model():
    def __init__(self,vocab_size,lr=1e-3,
                 embSize=128,
                 hidden_units = 64,
                 num_class = 2,
                 init_lr = 1e-3): 
        """
        kernel_size:3 as kernel
        do blocks: conv1d--conv1d--max_pooling  (out_filter*2)conv1d -->.......
        """
        with tf.variable_scope('inputs',reuse=tf.AUTO_REUSE):
            self.is_training = tf.placeholder(tf.bool,name='training_flag',
                                              shape=[])
            self.inputs = tf.placeholder(tf.int32,shape =[None,None],
                                         name='intputs')
            self.targets = tf.placeholder(tf.int32,shape=[None])
        
            self.keep_prob = tf.placeholder(tf.float32,shape=[],name='keep_prob')

            #global_step = tf.get_variable(name='gs',trainable=False,initializer=tf.zeros([]))

        emb_init_func = tf.random_uniform_initializer(-1.0,1.0)
        cnn_init_func = tf.truncated_normal_initializer(stddev=0.02)
        
        with tf.variable_scope('embedding',initializer=emb_init_func,reuse=tf.AUTO_REUSE):
            embedding = tf.get_variable(name='embTabel',shape=[vocab_size,embSize],dtype=tf.float32)
            def embedding_lookup(table,x):
                ntable = tf.concat([tf.zeros([1,embSize]),table[1:,:]],axis=0)
                return tf.nn.embedding_lookup(ntable,x)
        
        def conv1d(inputs,output_filters,training,vname):
            bn = tf.layers.batch_normalization(inputs,training=training,
                                               name = vname+'/bn')
            
            relu = tf.nn.leaky_relu(bn,0.2,name=vname+'/relu')

            conved = tf.layers.conv1d(inputs=relu,filters=output_filters,
                                      kernel_size=3,
                                      padding='same',
                                      kernel_initializer=cnn_init_func,
                                      activation = None,
                                      name = vname+'/conv1d'
                                      )
            return conved
        
        def max_pooling(inputs,vname):
            return tf.layers.max_pooling1d(inputs,pool_size=2,strides=2,
                                           padding='valid',name=vname+'/max_pooling')
        
        def global_pooling(inputs,vname):
            return tf.reduce_max(inputs,axis=[1],name=vname+'/global_pooling')
                

        embed_x = embedding_lookup(embedding,self.inputs)
        
        with tf.variable_scope('block1'):
            conv1 = conv1d(embed_x,32,self.is_training,vname='conv1')
            conv2 = conv1d(conv1,32,self.is_training,vname='conv2')
            pool3 = max_pooling(conv2,vname='pool3')
        with tf.variable_scope('block2'):
            conv4 = conv1d(pool3,64,self.is_training,vname='conv4')
            conv5 = conv1d(conv4,64,self.is_training,vname='conv5')
            pool6 = max_pooling(conv5,vname='pool6')
        with tf.variable_scope('block3'):
            conv7 = conv1d(pool6,128,self.is_training,vname='conv7')
        cnnOutput = tf.nn.dropout(global_pooling(conv7,vname='gp'),
                               keep_prob=self.keep_prob)
        with tf.variable_scope('fc'):
            fc1 = tf.layers.dense(cnnOutput,hidden_units,name='fc1',
                                  activation=tf.nn.leaky_relu)
            logits = tf.layers.dense(fc1,num_class,name='fc2',
                                  activation=None)
        with tf.variable_scope('learning_rate'):
            global_step = tf.Variable(0.0,name='gs')
            learning_rate = tf.train.exponential_decay(learning_rate = init_lr,
                                                       global_step = global_step,
                                                       decay_steps = 1000,
                                                       decay_rate = 0.9,
                                                       staircase=True,
                                                       name='learning_rate')

        with tf.variable_scope('metrics',reuse=tf.AUTO_REUSE):
            softmax_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits = logits,
                labels = self.targets
            )

            self.cost = tf.reduce_mean(softmax_loss)

            self.prediction = tf.argmax(logits,1,output_type = tf.int32)

            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.targets,self.prediction),
                                    tf.float32))

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.variable_scope('trainop',reuse=tf.AUTO_REUSE):
            with tf.control_dependencies(update_ops):               
                self.trainop = tf.train.AdamOptimizer(learning_rate).minimize(self.cost,global_step)
            
if __name__ == '__main__':
    model = Model(3500)
    print(model.cost)