'''
define net structure
'''
import tensorflow as tf 
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import math_ops
from config import IMAGE_SIZE
import os

class Model(object):
    def __init__(self,sess,param):
        self.step = 0
        self._session = sess
        self.is_training = True
        self.__learn_rate = param["learn_rate"]
        self.__max_to_keep=param["max_to_keep"]
        self.__checkPoint_dir = param["checkPoint_dir"]
        self.__restore = param["b_restore"]
        self.__mode= param["mode"]
        self.__batch_size = param["batch_size"]
        with self._session.as_default():
            self.build_model()
        with self._session.as_default():
            self.init_op.run()
            self._saver = tf.train.Saver(tf.global_variables(),max_to_keep=self.__max_to_keep)
            #load parameters
            if self.__restore:
                ckpt = tf.train.latest_checkpoint(self.__checkPoint_dir)
                if(ckpt):
                    self.step = int(ckpt.split('-')[1])
                    self._saver.restore(self._session,ckpt)
                    print('restoring from step {}'.format(self.step))
                    self.step += 1
    def build_model(self):
        def SegmentNet(input,scope,is_training,reuse=None):
            '''
            model of segmentnet
            '''
            with tf.variable_scope(scope,reuse=reuse):
                with slim.arg_scope([slim.conv2d],padding='SAME',\
                    activation_fn=tf.nn.relu,normalizer_fn=slim.batch_norm):
                    net = slim.conv2d(input,32,[5,5],scope='conv1')
                    net = slim.conv2d(net,32,[5,5],scope='conv2')
                    net = slim.max_pool2d(net,[2,2],scope='pool1')

                    net = slim.conv2d(net,64,[5,5],scope='conv3')
                    net = slim.conv2d(net,64,[5,5],scope='conv4')
                    net = slim.conv2d(net,64,[5,5],scope='conv5')
                    net = slim.max_pool2d(net,[2,2],scope='pool2')

                    net = slim.conv2d(net,64,[5,5],scope='conv6')
                    net = slim.conv2d(net,64,[5,5],scope='conv7')
                    net = slim.conv2d(net,64,[5,5],scope='conv8')
                    net = slim.conv2d(net,64,[5,5],scope='conv9')
                    net = slim.max_pool2d(net,[2,2],scope='pool3')

                    net = slim.conv2d(net,1024,[15,15],scope='conv10')
                    features = net #input featrues in decision net
                    net = slim.conv2d(net,1,[1,1],activation_fn=None,scope='conv11')
                    logits_pixel = net
                    mask = tf.sigmoid(net,name=None)
            return features,logits_pixel,mask
        
        def DecisionNet(featrues,mask,scope,is_training,num_class=2,reuse=None):
            '''
            model of decision
            '''
            with tf.variable_scope(scope,reuse=None):
                with slim.arg_scope([slim.conv2d],padding='SAME',\
                    activation_fn=tf.nn.relu,normalizer_fn=slim.batch_norm):
                    net = tf.concat([featrues,mask],axis=3)
                    net = slim.max_pool2d(net,[2,2],scope='pool1')

                    net = slim.conv2d(net,8,[5,5],scope='conv1')
                    net = slim.max_pool2d(net,[2,2],scope='pool2')
                    net = slim.conv2d(net,16,[5,5],scope='conv2')
                    net = slim.max_pool2d(net,[2,2],scope='pool3')

                    net = slim.conv2d(net,32,[5,5],scope='conv3')

                    vector1 = tf.reduce_mean(net,axis=[1,2],name='pool4',keepdims=True)
                    vector2 = tf.reduce_max(net,axis=[1,2],name='pool5',keepdims=True)
                    vector3 = tf.reduce_mean(mask,axis=[1,2],name='pool6',keepdims=True)
                    vector4 = tf.reduce_max(mask,axis=[1,2],name='pool7',keepdims=True)
                    vector = tf.concat([vector1,vector2,vector3,vector4],axis=3)
                    vector = tf.squeeze(vector,[1,2])
                    logits = slim.fully_connected(vector,num_class,activation_fn=None)
                    output = tf.argmax(logits,axis=1)
            return logits,output

        Image = tf.placeholder(tf.float32, shape=(self.__batch_size, IMAGE_SIZE[0],IMAGE_SIZE[1], 1), name='Image')
        PixelLabel=tf.placeholder(tf.float32, shape=(self.__batch_size, IMAGE_SIZE[0]/8,IMAGE_SIZE[1]/8, 1), name='PixelLabel')
        Label = tf.placeholder(tf.int32, shape=(self.__batch_size), name='Label')
        features, logits_pixel, mask=SegmentNet(Image,'segment',self.is_training)
        logits_class,output_class=DecisionNet(features,mask, 'decision', self.is_training)
        #define loss function
        logits_pixel=tf.reshape(logits_pixel,[self.__batch_size,-1])
        PixelLabel_reshape=tf.reshape(PixelLabel,[self.__batch_size,-1])
        loss_pixel = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_pixel, labels=PixelLabel_reshape))
        loss_class = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_class,labels=Label))
        loss_total=loss_pixel+loss_class
        optimizer = tf.train.GradientDescentOptimizer(self.__learn_rate)
        train_var_list = [v for v in tf.trainable_variables() ]
        train_segment_var_list = [v for v in tf.trainable_variables() if 'segment' in v.name ]
        train_decision_var_list = [v for v in tf.trainable_variables() if 'decision' in v.name]
        optimize_segment = optimizer.minimize(loss_pixel,var_list=train_segment_var_list)
        optimize_decision = optimizer.minimize(loss_class, var_list=train_decision_var_list)
        optimize_total = optimizer.minimize(loss_total, var_list=train_var_list)
        init_op=tf.global_variables_initializer()
        self.Image=Image
        self.PixelLabel = PixelLabel
        self.Label = Label
        self.features = features
        self.mask = mask
        self.logits_class=logits_class
        self.output_class=output_class
        self.loss_pixel = loss_pixel
        self.loss_class = loss_class
        self.loss_total = loss_total
        self.optimize_segment = optimize_segment
        self.optimize_decision = optimize_decision
        self.optimize_total = optimize_total
        self.init_op=init_op
    
    def save(self):
        '''
        save the model
        '''
        self._saver.save(self._session,os.path.join(self.__checkPoint_dir,'ckp'),\
            global_step=self.step)
                