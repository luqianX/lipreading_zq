#!/usr/bin/env python
# -*- coding: utf-8 -*-
#================================================================
#   God Bless You.
#
#   file name: cnn_extractor.py
#   author: klaus
#   email: klaus.cheng@qq.com
#   created date: 2018/07/10
#   description:
#
#================================================================

import tensorflow as tf
import tensorflow.contrib.keras as keras



class CNN(object):
    """base cnn model. Extract feature of the video_tensor.

    Input:
        video_tensor: Tensor. videos of shape (batch_size, T, H, W, C). In GRID. H = 50 and W = 100.
    Output: Tensor of shape(T, feature_len)此处是否少了batch_size

    """

    def __init__(self, feature_len, training, scope='cnn_feature_extractor'):
        self.feature_len = feature_len
        self.training = training
        self.scope = scope

    def build():
        raise NotImplementedError('CNN not NotImplemented.')


class EarlyFusion2D(CNN):
    """early fusion + 2D cnn"""

    def __init__(self, *args, **kwargs):
        super(EarlyFusion2D, self).__init__(*args, **kwargs)

    def build(self, video_tensor):
        """build model.

        Args:
            video_tensor: Tensor. videos of shape (batch_size, T, H, W, C). In GRID. H = 50 and W = 100.

        Returns: the output tensor of the model

        """
        with tf.variable_scope(self.scope,reuse=True):
            #(32, 75, 50, 100, 3)
            self.conv1 = keras.layers.Conv3D(
                32, (5, 5, 5),
                strides=(1, 2, 2),
                padding='same',
                #kernel_initializer='he_normal',
                kernel_initializer=initializers.he_normal(seed=1024),
                name='conv1')(video_tensor)
            #(32, 75, 25, 50, 32)
            self.batc1 = tf.layers.batch_normalization(
                self.conv1, training=self.training, name='batc1')
            self.actv1 = keras.layers.Activation(
                'relu', name='actv1')(self.batc1)
            self.drop1 = keras.layers.SpatialDropout3D(0.5)(self.actv1)
            self.maxp1 = keras.layers.MaxPooling3D(
                pool_size=(1, 2, 2), strides=(1, 2, 2),
                name='max1')(self.drop1)
            #(32, 75, 12, 25, 32)
            self.conv2 = keras.layers.TimeDistributed(
                keras.layers.Conv2D(
                    64, (5, 5),
                    strides=(1, 1),
                    padding='same',
                    kernel_initializer='he_normal'),
                name="TD_conv2")(self.maxp1)
            #(32, 75, 12, 25, 64)
            self.batc2 = tf.layers.batch_normalization(
                self.conv2, training=self.training, name='batc2')
            self.actv2 = keras.layers.Activation(
                'relu', name='actv2')(self.batc2)
            self.drop2 = keras.layers.SpatialDropout3D(0.5)(self.actv2)
            self.maxp2 = keras.layers.MaxPooling3D(
                pool_size=(1, 2, 2), strides=(1, 2, 2),
                name='max2')(self.drop2)
            #(32, 75, 6, 12, 64)
            self.conv3 = keras.layers.TimeDistributed(
                keras.layers.Conv2D(
                    96, (3, 3),
                    strides=(1, 1),
                    padding='same',
                    kernel_initializer='he_normal'),
                name="TD_conv3")(self.maxp2)
            #(32, 75, 6, 12, 96)
            self.batc3 = tf.layers.batch_normalization(
                self.conv3, training=self.training, name='batc3')
            self.actv3 = keras.layers.Activation(
                'relu', name='actv3')(self.batc3)
            self.drop3 = keras.layers.SpatialDropout3D(0.5)(self.actv3)
            self.maxp3 = keras.layers.MaxPooling3D(
                pool_size=(1, 2, 2), strides=(1, 2, 2),
                name='max3')(self.drop3)
            #(32, 75, 3, 6, 96)
            # prepare output
            self.conv4 = keras.layers.Conv3D(
                self.feature_len, (1, 1, 1),
                strides=(1, 1, 1),
                kernel_initializer='he_normal',
                name='conv4')(self.maxp3)
            #(32, 75 ,3 , 6, 512)
            self.output = keras.layers.TimeDistributed(
                # keras.layers.GlobalAveragePooling2D(name='global_ave1'),
                keras.layers.GlobalMaxPool2D(name='global_ave1'),
                name='TD_GMP1')(self.conv4)  #shape: (T, feature_len)
            #(32, 75 , 512)
            return self.output


class LipNet(CNN):
    """lipnet cnn feature extractor"""

    def __init__(self, *args, **kwargs):
        super(LipNet, self).__init__(*args, **kwargs)

    def build(self, video_tensor):
        """build model.

        Args:
            video_tensor: Tensor. videos of shape (batch_size, T, H, W, C). In GRID. H = 50 and W = 100.

        Returns: the output tensor of the model

        """
        with tf.variable_scope(self.scope,reuse=tf.AUTO_REUSE):
            #(32, 75, 50, 100, 3)
            self.zero1 = keras.layers.ZeroPadding3D(
                padding=(1, 2, 2), name='zero1')(video_tensor)
            #(32, 77, 54, 104, 3)
            self.conv1 = keras.layers.Conv3D(
                32, (3, 5, 5),
                strides=(1, 2, 2),
                #kernel_initializer='he_normal',
                kernel_initializer=keras.initializers.he_normal(seed=1024),
                bias_initializer=tf.constant_initializer(0),
                trainable=self.training,
                name='conv1')(self.zero1)
            #(32, 75, 25, 50, 32)
            # self.batc1 = keras.layers.BatchNormalization(name='batc1')(
            # self.conv1, training=self.training)
            self.batc1 = tf.layers.batch_normalization(
                self.conv1, 
                training=self.training, 
                #trainable=False,
                #training=True,
                name='batc1')
            self.actv1 = keras.layers.Activation(
                'relu', name='actv1')(self.batc1)
#                'relu', name='actv1')(self.conv1)
            self.drop1 = keras.layers.SpatialDropout3D(0.5)(self.actv1)
            self.maxp1 = keras.layers.MaxPooling3D(
                pool_size=(1, 2, 2), strides=(1, 2, 2),
                name='max1')(self.drop1)
            #(32, 75, 12, 25, 32)
            self.zero2 = keras.layers.ZeroPadding3D(
                padding=(1, 2, 2), name='zero2')(self.maxp1)
            #(32, 77, 16, 29, 32)
            self.conv2 = keras.layers.Conv3D(
                64, (3, 5, 5),
                strides=(1, 1, 1),
                #kernel_initializer='he_normal',
                kernel_initializer=keras.initializers.he_normal(seed=1024),
                bias_initializer=tf.constant_initializer(0),
                trainable=self.training,
            #    training=self.training,
                name='conv2'
                )(self.zero2)
            #(32, 75, 12, 25, 64)
            # self.batc2 = keras.layers.BatchNormalization(name='batc2')(
            # self.conv2, training=self.training)
            self.batc2 = tf.layers.batch_normalization(
                self.conv2, 
                training=self.training,
            #    training=True,
            #    trainable=False,
                name='batc2')
            self.actv2 = keras.layers.Activation(
                'relu', name='actv2')(self.batc2)
            self.drop2 = keras.layers.SpatialDropout3D(0.5)(self.actv2)
            self.maxp2 = keras.layers.MaxPooling3D(
                pool_size=(1, 2, 2), strides=(1, 2, 2),
                name='max2')(self.drop2)
            #(32, 75, 6, 12, 64)
            self.zero3 = keras.layers.ZeroPadding3D(
                padding=(1, 1, 1), name='zero3')(self.maxp2)
            #(32, 77, 8, 14, 64)
            self.conv3 = keras.layers.Conv3D(
                256,
            #    96,
                (3, 3, 3),
                strides=(1, 1, 1),
                #kernel_initializer='he_normal',
                kernel_initializer=keras.initializers.he_normal(seed=1024),
                bias_initializer=tf.constant_initializer(0),
                trainable=self.training,
                name='conv3')(self.zero3)
            #(32, 75, 6, 12, 256)
            # self.batc3 = keras.layers.BatchNormalization(name='batc3')(
            # self.conv3, training=self.training)
            self.batc3 = tf.layers.batch_normalization(
                self.conv3, 
                training=self.training,
            #    trainable=False,
                name='batc3')
            self.actv3 = keras.layers.Activation(
                'relu', name='actv3')(self.batc3)
            self.drop3 = keras.layers.SpatialDropout3D(0.5)(self.actv3)
            self.maxp3 = keras.layers.MaxPooling3D(
                pool_size=(1, 2, 2), strides=(1, 2, 2),
                name='max3')(self.drop3)
            #(32, 75, 3, 6, 256)
     #       self.x_on=keras.layers.Lambda(lambda x:x[:,:,:,:,0:48])(self.maxp3)
     #       self.x_down=keras.layers.Lambda(lambda x:x[:,:,:,:,48:96])(self.maxp3)

     #       self.x_on_w=keras.layers.Lambda(lambda x:0.1*x)(self.x_on)
     #       self.x_down_w=keras.layers.Lambda(lambda x:0.9*x)(self.x_down)
            
     #       self.concatenate=keras.layers.concatenate([self.x_on_w,self.x_down],axis=0)
     #       self.added = keras.layers.Add()([self.x_down_w, self.x_on_w])
            # prepare output
            self.conv4 = keras.layers.Conv3D(
            #    self.feature_len, 
            #    1024,
                512,
                (1, 1, 1),
                strides=(1, 1, 1),
                #kernel_initializer='he_normal',
                trainable=self.training,
                kernel_initializer=keras.initializers.he_normal(seed=1024),
                bias_initializer=tf.constant_initializer(0),
            #    name='conv4')(self.x_down)
                name='conv4')(self.maxp3)
            #    name='conv4')(self.concatenate)
            #    name='conv4')(self.added)
            #(32, 75, 3, 6, 512)
            self.x_on=keras.layers.Lambda(lambda x:x[:,:,:,:,0:256])(self.conv4)
            self.x_down=keras.layers.Lambda(lambda x:x[:,:,:,:,256:512])(self.conv4)
            #两个(32, 75, 3, 6, 256)
            #全局最大池化
            self.output = keras.layers.TimeDistributed(
                keras.layers.GlobalMaxPooling2D(name='global_ave1'),
                name='timeDistributed1')(self.x_down)  #shape: (T, feature_len)
            #(32, 75, 256)
         #   self.output=keras.layers.concatenate([self.x_on,self.x_down],axis=-1)
         #   self.output = keras.layers.Add()([self.x_down_w, self.x_on_w])    
         #   print (self.feature_len)
            return self.output,self.x_on,self.x_down
        #    return self.output,self.x_on,self.x_mid,self.x_down
   
