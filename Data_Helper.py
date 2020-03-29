'''
class Data_Helper offer a mean
of managing data
'''
import numpy as np 
from cv2 import cv2
import os
from config import IMAGE_SIZE
import tensorflow as tf 

class Data_Helper(object):
    def __init__(self,data_list,param,shuffle=True):
        '''
        datalist : image and its label image's dir

        param : train parameter you set

        shuffle : decide if data will be shuffled 
        '''
        self.shuffle = shuffle
        self.data_list = data_list
        self.data_size = len(data_list)
        self.data_dir = param['data_dir']
        self.epochs_num = param['epochs_num']
        self.batch_size = param['batch_size']
        self.steps = int(np.floor(self.data_size / self.batch_size))#training step per epoch
        self.next_batch = self.get_next()

    def get_next(self):
        '''
        return a batch of data consist of
        image and processed image and label and image's full path
        '''
        dataset = tf.data.Dataset.from_generator(self.data_generator,\
            (tf.float32,tf.float32,tf.int32,tf.string))
        dataset = dataset.repeat(self.epochs_num)
        if self.shuffle:
            dataset.shuffle(self.batch_size*200)
        dataset = dataset.batch(self.batch_size)

        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()

    def data_generator(self):
        '''
        data generator called by func get_next()
        yield image and processed image and label and image's full path
        image:input image in the net

        label_image:label image which is reiszed

        label:is label image with defect

        image_fullpath:absolute path of input image
        '''
        for image_path,label_path in self.data_list:
            image_fullpath = os.path.join(self.data_dir,image_path)
            label_fullpath = os.path.join(self.data_dir,label_path)
            image = self.read_image(image_fullpath)
            label = self.read_image(label_fullpath)
            label_image,label = self.label_process(label)
            image = np.array(image[:,:,np.newaxis]) #add third dim
            label_image = np.array(label_image[:,:,np.newaxis])
            yield image,label_image,label,image_path
    
    def read_image(self,path):
        '''
        read the image and resized in asked size
        '''
        image = cv2.imread(path,0)# read image in gray scale,result is a m*n matrix
        image = cv2.resize(image,(IMAGE_SIZE[1],IMAGE_SIZE[0]))
        return image

    def label_process(self,label):
        '''
        resized the label image and decide
        if it is a defected image
        '''
        label_image = cv2.resize(label,(int(IMAGE_SIZE[1]/8), int(IMAGE_SIZE[0]/8)))
        label = self.is_defected(label)
        return label_image,label
    
    def is_defected(self,label,threshold=1):
        '''
        decide is the label image with defected,consider that 
        a image without defect,its label image is pure black, so pixel sum is 0
        '''
        label = np.array(label)
        label = np.where(label > threshold,1,0)
        label = label.sum()
        if(label > 0):
            label = 1
        return label



