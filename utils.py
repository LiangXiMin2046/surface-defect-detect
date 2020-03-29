import logging
import os
import time
from PIL import Image

def get_logger(log_path='log_path'):
    '''
    help log daily
    '''
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    timer = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime())
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(levelname)s]   %(asctime)s    %(message)s')
    handler = logging.FileHandler((log_path+'/'+timer+'log.txt'))
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def concatImage(images,mode='L'):
    '''
    concatenate original image and 
    label image and mask image
    '''
    if not isinstance(images,list):
        raise Exception('images must be a list')
    count = len(images)
    size = Image.fromarray(images[0]).size
    target = Image.new(mode,(size[0] * count,size[1]))
    for i in range(count):
        image = Image.fromarray(images[i]).resize(size,Image.BILINEAR)
        target.paste(image,(i*size[0],0,(i+1)*size[0],size[1]))
    return target
     