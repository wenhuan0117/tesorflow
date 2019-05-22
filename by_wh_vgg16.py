import os
import numpy as np
import tensorflow as tf
import skimage.io
import skimage.transform
import matplotlib.pyplot as plt
import cv2
import readimages as rd

learning_rate=0.001
num_steps=20
batch_size=20
disp_step=2
dropout=0.75

N_CLASSES=40

npy_path='G:/0program/python/vgg16.npy'
train_data_path='G:/0program/python/opencv_test/face_recog/face_data'
save_model_path="./my_net/save_net"
pre_image_path='G:/0program/python/opencv_test/face_recog/face_data/Abdullah/Abdullah_0001.jpg'

##########################
def load_image(image_path):
    
    img=cv2.imread(image_path)
    img=img/255.0
    image=cv2.resize(img,(224,224))
    xs=[]
    xs.append(image)
    return xs


################################
def creat_train_data(data_path):
    
    imagepaths,labels=rd.read_images(data_path)
    x_data,y_data,ran_list=rd.load_images(batch_size,imagepaths,labels)

    with open("test.txt","w") as f:
        for i in ran_list:
            f.write(imagepaths[i]+'  ')
            f.write(imagepaths[i].split('\\')[-2]+'  ')
            f.write(str(labels[i]))
            f.write('\n')
    return x_data,y_data



####################################
def build_vgg16(tfx):
    
    data_dict=np.load(npy_path,encoding='latin1').item()
    vgg_mean=[103.939,116.779,123.68]
    red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=tfx * 255.0)
    bgr = tf.concat(axis=3, values=[
        blue -vgg_mean[0],
        green - vgg_mean[1],
        red -vgg_mean[2],
    ])

    def max_pool(bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(bottom, name):
        with tf.variable_scope(name):   # CNN's filter is constant, NOT Variable that can be trained
            conv = tf.nn.conv2d(bottom, data_dict[name][0], [1, 1, 1, 1], padding='SAME')
            lout = tf.nn.relu(tf.nn.bias_add(conv, data_dict[name][1]))
            return lout
    conv1_1 =conv_layer(bgr, "conv1_1")
    conv1_2 =conv_layer(conv1_1, "conv1_2")
    pool1 = max_pool(conv1_2, 'pool1')

    conv2_1 =conv_layer(pool1, "conv2_1")
    conv2_2 =conv_layer(conv2_1, "conv2_2")
    pool2 = max_pool(conv2_2, 'pool2')

    conv3_1 =conv_layer(pool2, "conv3_1")
    conv3_2 =conv_layer(conv3_1, "conv3_2")
    conv3_3 =conv_layer(conv3_2, "conv3_3")
    pool3 = max_pool(conv3_3, 'pool3')

    conv4_1 =conv_layer(pool3, "conv4_1")
    conv4_2 =conv_layer(conv4_1, "conv4_2")
    conv4_3 =conv_layer(conv4_2, "conv4_3")
    pool4 = max_pool(conv4_3, 'pool4')

    conv5_1 =conv_layer(pool4, "conv5_1")
    conv5_2 =conv_layer(conv5_1, "conv5_2")
    conv5_3 =conv_layer(conv5_2, "conv5_3")
    pool5 = max_pool(conv5_3, 'pool5')

    flatten = tf.reshape(pool5, [-1, 7*7*512])
    fc6 = tf.layers.dense( flatten, 50, tf.nn.relu, name='fc6')
    out = tf.layers.dense( fc6, N_CLASSES, name='out')
    return out

##########################################
def train_op(tfx,x_data,y_data,save_model_path):
    
    out=build_vgg16(tfx)
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits( 
            logits=out, labels=y_data))
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss_op)

    saver=tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(num_steps):
            los,_=sess.run([loss_op,train_op],feed_dict={tfx:x_data})
            if i%disp_step==0:
                print(i,los)
            
            save_path=saver.save(sess,save_model_path)
            saver.save(sess,save_path)
            
##################################            
def train_net(train_data_path,save_model_path):

    tfx=tf.placeholder(tf.float32,[None,224,224,3])    
    x_data,y_data=creat_train_data(train_data_path)    
    global N_CLASSES
    N_CLASSES=max(y_data)+1
    
    train_op(tfx,x_data,y_data,save_model_path)
    
######################    
def prediction_net(image_path,save_model_path):

    x_data=load_image(image_path)
    tfx=tf.placeholder(tf.float32,[None,224,224,3])
    out=build_vgg16(tfx)
    pre=tf.argmax(tf.nn.softmax(out),1)
    saver=tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess,save_model_path)
        print(sess.run(pre,feed_dict={tfx:x_data}))

##train_net(train_data_path,save_model_path)

prediction_net(pre_image_path,save_model_path)
