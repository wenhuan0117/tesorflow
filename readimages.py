import tensorflow as tf
import os
import numpy as np
import cv2



image_h=224
image_w=224
image_channel=3


def read_images(data_path):
    imagepaths,labels=list(),list()
    label=0
    classes=sorted(os.walk(data_path).__next__()[1])
    for c in classes:
        c_dir=os.path.join(data_path,c)
        walk=os.walk(c_dir).__next__()
        for sample in walk[2]:
            if sample.endswith('.jpg'):
                imagepaths.append(os.path.join(c_dir,sample))
                labels.append(label)
        label+=1
    return imagepaths,labels
        
def load_images(batch_size,imagepaths,labels):
    
    length=len(labels)
    a=range(length)
    ran_list=np.random.choice(a,batch_size,replace=False)
    x_batch=[]
    y_batch=[]

    for i in ran_list:
        y_batch.append(labels[i])
        img=cv2.imread(imagepaths[i])
        img=img/255.0
        image=cv2.resize(img,(image_h,image_w))
        x_batch.append(image)
    y_batch=np.array(y_batch)
##    y_batch=y_batch.reshape(batch_size,1)
        
    return x_batch,y_batch,ran_list

##data_path='G:/0program/python/opencv_test/lfw/data/lfw/'
##read_images(data_path)
##x,y=load_images(10)
##cv2.imshow('picture1',x[1])
##cv2.waitKey(0)
