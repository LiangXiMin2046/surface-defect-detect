'''
define train and test function
and data processing function
'''
from config import POSITIVE_KolektorSDD
import os
import numpy as np 
import tensorflow as tf
from Data_Helper import Data_Helper
from model import Model
import utils
from utils import concatImage

class Agent(object):
    def __init__(self,param,):
        self._session = tf.Session()
        self._param = param
        self.init_data()
        self.model = Model(self._session,self._param)
        self.logger = utils.get_logger(self._param['Log_dir'])
    def run(self):
        '''
        launch training or testing
        '''
        if self._param['mode'] is 'training':
            train_mode = self._param['train_mode']
            self.train(train_mode)
        elif self._param['mode'] is 'testing':
            self.test()
        else:
            raise Exception('unexcepted mode')
        
    def train(self,mode):
        '''
        train the model
        '''
        if mode not in ['segment','decision','total']:
            raise Exception('not in right train mode')
        with self._session.as_default():
            self.logger.info('start training {} net'.format(mode))
            for i in range(self.model.step,self._param['epochs_num'] + self.model.step):
                loss = 0
                for step in range(0,self.positive_train_datalist.steps):
                    for id in range(0,2):
                        if(id == 0):
                            image_batch,label_pixel_batch,label_batch,filename_batch = \
                                self._session.run(self.positive_train_datalist.next_batch)
                        elif(id == 1):
                            image_batch,label_pixel_batch,label_batch,filename_batch = \
                                self._session.run(self.negative_train_datalist.next_batch)
                        loss_step = 0
                        if mode is 'segment':
                            _,loss_step = self._session.run([self.model.optimize_segment,self.model.loss_pixel],\
                                feed_dict={self.model.Image:image_batch,self.model.PixelLabel:label_pixel_batch})
                        elif mode is 'decision':
                            _,loss_step = self._session.run([self.model.optimize_decision,self.model.loss_class],\
                                feed_dict={self.model.Image:image_batch,self.model.Label:label_batch})
                        elif mode is 'total':
                            _,loss_step = self._session.run([self.model.optimize_total,self.model.loss_total],\
                                feed_dict={self.model.Image:image_batch,self.model.PixelLabel:label_pixel_batch,self.model.Label:label_batch})
                        loss += loss_step
                self.logger.info('epoch:{} train mode{} loss{}'.format(self.model.step,mode,loss))
                if i == self._param['epochs_num'] + self.model.step - 1:
                    self.model.save()
                self.model.step += 1
    def test(self):
        visualization_dir = './visualization/test'
        if not os.path.exists(visualization_dir):
            os.makedirs(visualization_dir)
        with self._session.as_default():
            self.logger.info('start testing')
            count = 0
            count_TP = 0
            count_FP = 0
            count_TN = 0
            count_FN = 0
            data = [self.positive_test_datalist,self.negative_test_datalist]
            sample = 1
            for i in range(0,2):
                for step in range(data[i].steps):
                    image_batch,pixel_label_batch,label_batch,filename_batch = \
                        self._session.run(data[i].next_batch)
                    masks_batch,output_batch = \
                    self._session.run([self.model.mask,self.model.output_class],feed_dict=\
                        {self.model.Image:image_batch,})
                    self.visualization(image_batch,masks_batch,pixel_label_batch,filename_batch,visualization_dir)
                    for i,filename in enumerate(filename_batch):
                        count += 1
                        if label_batch[i] == 1 and output_batch[i] == 1:
                            count_TP += 1 #缺陷样本判定有缺陷
                        elif label_batch[i] == 1 and output_batch[i] == 0:
                            count_FP += 1 #缺陷样本判定无缺陷
                        elif label_batch[i] == 0 and output_batch[i] == 0:
                            count_TN += 1 #无缺陷样本判定无缺陷
                        else:
                            count_FN += 1 #无缺陷判定有缺陷
                        print('sameple {} has checked'.format(sample))
                        sample += 1
            print('done')
            accuracy = (count_TP + count_TN) / count
            precision = count_TP / (count_TP + count_FP)
            self.logger.info('samples:{}'.format(count))
            self.logger.info('count_TP:{}'.format(count_TP))
            self.logger.info('count_FP:{}'.format(count_FP))
            self.logger.info('count_TN:{}'.format(count_TN))
            self.logger.info('count_FN:{}'.format(count_FN))
            self.logger.info('accurary:{:.4f}'.format(accuracy))
            self.logger.info('precision:{:.4f}'.format(precision))
    
    def visualization(self,image_batch,masks_batch,pixel_label_batch,file_names,save_dir='./visualization'):
        '''
        visualize 
        '''
        for i,filename in enumerate(file_names):
            filename = str(filename).split("'")[-2].replace('/','_')
            mask = np.array(masks_batch[i]).squeeze(2) * 255
            image = np.array(image_batch[i]).squeeze(2)
            pixel_label = np.array(pixel_label_batch[i]).squeeze(2) * 255
            combination = concatImage([image,pixel_label,mask])
            combination.save(os.path.join(save_dir,filename))

    def init_data(self):
        '''
        prepare data
        '''
        self.positive_datalist,self.negative_datalist = \
            self.get_data(self._param['data_dir'])
        if self._param['mode'] is 'training':
            self.positive_train_datalist = Data_Helper(self.positive_datalist,self._param)
            self.negative_train_datalist = Data_Helper(self.negative_datalist,self._param)
        elif self._param['mode'] is 'testing':
            self.positive_test_datalist = Data_Helper(self.positive_datalist,self._param)
            self.negative_test_datalist = Data_Helper(self.negative_datalist,self._param)
        else:
            raise Exception('unexcepted mode')
    def get_data(self,data_dir,test_ration=0.4,positive_index=POSITIVE_KolektorSDD):
        '''
        return a list contains pairs of 
        image and its label's path
        '''
        example_dirs = os.listdir(data_dir)
        example_lists = {x : os.listdir(os.path.join(data_dir,x)) for x in example_dirs}
        positive_train_examples = []
        negative_train_examples = []
        positive_val_examples = []
        negative_val_examples = []
        offset = len(example_dirs) * (1 - test_ration)
        for i in range(len(example_dirs)):
            example_dir = example_dirs[i]
            example_list = example_lists[example_dir]
            example_list = [x for x in example_list if 'label' not in x]
            if i < offset:
                for image in example_list:
                    example_image = example_dir + '/' + image
                    label_image = example_image.split('.')[0] + '_label.bmp'
                    idx = image.split('.')[0][-1]
                    if idx in positive_index[i]:
                        positive_train_examples.append([example_image,label_image])
                    else:
                        negative_train_examples.append([example_image,label_image])
            else:
                for image in example_list:
                    example_image = example_dir + '/' + image
                    label_image = example_image.split('.')[0] + '_label.bmp'
                    idx = image.split('.')[0][-1]
                    if idx in positive_index[i]:
                        positive_val_examples.append([example_image,label_image])
                    else:
                        negative_val_examples.append([example_image,label_image])
        if self._param['mode'] is 'training':
            return positive_train_examples,negative_train_examples
        if self._param['mode'] is 'testing':
            return positive_val_examples,negative_val_examples
            
                
                