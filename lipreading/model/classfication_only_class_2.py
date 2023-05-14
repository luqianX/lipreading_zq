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

    def __init__(self, training, scope='Classfication'):
#        self.feature_len = feature_len
        self.training = training
        self.scope = scope

    def build():
        raise NotImplementedError('CNN not NotImplemented.')





class Classfication(CNN):
    """classfication"""

    def __init__(self, *args, **kwargs):
        super(Classfication, self).__init__(*args, **kwargs)

    def build(self, x_on):
        """build model.

        Args:
            video_tensor: Tensor. videos of shape (batch_size, T, H, W, C). In GRID. H = 50 and W = 100.

        Returns: the output tensor of the model

        """
        with tf.variable_scope(self.scope,reuse=tf.AUTO_REUSE):
            #(32,75,3,6,256)
            self.output = keras.layers.TimeDistributed(
                keras.layers.GlobalMaxPooling2D(name='global_ave1'),
                name='timeDistributed1')(x_on) 
            #(32,75,256)
            self.gap=keras.layers.GlobalAveragePooling1D(name='GAP2D')(self.output)
            #(32,256)
            self.fc1=keras.layers.Dense( 512,name='fc1',
                                        trainable=self.training,
                                        )(self.gap)
            #(32,512)
            self.drop1=keras.layers.Dropout(0.5)(self.fc1)
            self.out=keras.layers.Dense( 45,name='fc2',
                                        trainable=self.training,
                                        )(self.drop1)
            #(32,29)
            self.softmax=keras.layers.Activation('softmax')(self.out)
        #    self.out=keras.layers.Dense( 33,name='fc',trainable=self.training,)(self.flat)
        #    self.out=keras.layers.Dense( 33,activation='softmax',name='fc',trainable=self.training,)(self.flat)
            return self.out,self.softmax
