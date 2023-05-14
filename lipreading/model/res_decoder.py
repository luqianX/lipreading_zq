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
    Output: Tensor of shape(T, feature_len)

    """

    def __init__(self, training, scope='3D_decoder'):
#        self.feature_len = feature_len
        self.training = training
        self.scope = scope

    def build():
        raise NotImplementedError('CNN not NotImplemented.')





class Decoder(CNN):
    """3D_deocder"""

    def __init__(self, *args, **kwargs):
        super(Decoder, self).__init__(*args, **kwargs)

    def build(self, x_on,x_down):
#    def build(self, x_on,x_mid,x_down):
        """build model.

        Args:
            video_tensor: Tensor. videos of shape (batch_size, T, H, W, C). In GRID. H = 50 and W = 100.

        Returns: the output tensor of the model

        """
        with tf.variable_scope(self.scope,reuse=tf.AUTO_REUSE):
            #(32,75,3,6,256);(32,75,3,6,256)
            self.concatenate=keras.layers.concatenate([x_on,x_down],axis=-1)
            #(32,75,3,6,512)
        #    self.concatenate=keras.layers.concatenate([x_on,x_mid,x_down],axis=-1)
        
        
            self.upsample1 = keras.layers.UpSampling3D(size=(1,2,2), 
                                                    #   input_shape=(75,3,6,96)
                                                       )(self.concatenate)
            #(32,75,6,12,512)
            self.conv1 = keras.layers.Conv3D  (
                           filters=64,
                           kernel_size=(3, 3, 3),
                        #   kernel_initializer=keras.initializers.he_normal(seed=1024),
                           strides=(1, 1, 1),
                           name="conv1", 
                           trainable=self.training,
                           padding='same')(self.upsample1)
            #(32,75,6,12,64)
            # self.batc1 = keras.layers.BatchNormalization(name='batc1')(
            # self.conv1, training=self.training)
            self.batc1 = tf.layers.batch_normalization(
                self.conv1, 
                training=self.training,
                name='batc1')
#            self.batc1 = keras.layers.BatchNormalization(
                #self.conv1, 
                #training=self.training, 
                #trainable=False,
#                name='batc1')(self.conv1)
            self.actv1 = keras.layers.Activation(
                'relu', name='actv1')(self.batc1)
            self.drop1 = keras.layers.SpatialDropout3D(0.5)(self.actv1)

            self.upsample2 = keras.layers.UpSampling3D(size=(1,2,2))(self.drop1)
            #(32,75,12,24,64)
            self.conv2 = keras.layers.Conv3D  (filters=32,
                           kernel_size=(3, 5, 5),
                        #   kernel_initializer=keras.initializers.he_normal(seed=1024),
                           strides=(1, 1, 1),
                           name="conv2", 
                           trainable=self.training,
                           padding='same')(self.upsample2)
            # self.batc1 = keras.layers.BatchNormalization(name='batc1')(
            # self.conv1, training=self.training)
            #(32,75,12,24,32)
            self.batc2 = tf.layers.batch_normalization(
                self.conv2, 
                training=self.training,
                name='batc2')
            self.actv2 = keras.layers.Activation(
                'relu', name='actv2')(self.batc2)
            self.drop2 = keras.layers.SpatialDropout3D(0.5)(self.actv2)
            
            self.upsample3 = keras.layers.UpSampling3D(size=(1,2,2))(self.drop2)
            #(32,75,24,48,32)
            self.time1=keras.layers.TimeDistributed(keras.layers.ZeroPadding2D(padding=(0,1)))(self.upsample3)
            #(32,75,24,50,32)
            self.conv3 = keras.layers.Conv3D  (filters=16,
                           kernel_size=(3, 5, 5),
                        #   kernel_initializer=keras.initializers.he_normal(seed=1024),
                           strides=(1, 1, 1),
                           name="conv3", 
                           trainable=self.training,
                           padding='same')(self.time1)
            #(32,75,24,50,16)
            # self.batc1 = keras.layers.BatchNormalization(name='batc1')(
            # self.conv1, training=self.training)
            self.batc3 = tf.layers.batch_normalization(
                self.conv3, 
                training=self.training,
                name='batc3')
#            self.batc3 = keras.layers.BatchNormalization(
#                name='batc3')(self.conv3)
            self.actv3 = keras.layers.Activation(
                'relu', name='actv3')(self.batc3)
            self.drop3 = keras.layers.SpatialDropout3D(0.5)(self.actv3)

            self.upsample4 = keras.layers.UpSampling3D(size=(1,2,2))(self.drop3)
            #(32,75,48,100,16)
            self.time2=keras.layers.TimeDistributed(keras.layers.ZeroPadding2D(padding=(1,0)))(self.upsample4)
            #(32,75,50,100,16)
            self.conv4 = keras.layers.Conv3D  (filters=3,
                           kernel_size=(3, 3, 3),
                        #   kernel_initializer=keras.initializers.he_normal(seed=1024),
                           strides=(1, 1, 1),
                           name="conv4", 
                           trainable=self.training,
                           padding='same')(self.time2)
            #(32,75,50,100,3)
            # self.batc1 = keras.layers.BatchNormalization(name='batc1')(
            # self.conv1, training=self.training)
            self.batc4 = tf.layers.batch_normalization(
                self.conv4, 
                training=self.training,
                name='batc4')
            self.actv4 = keras.layers.Activation(
                'sigmoid', name='actv3')(self.batc4)

            return self.actv4
