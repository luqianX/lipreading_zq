# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Transformer model helper methods."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf

_NEG_INF = -1e9


def get_position_encoding(length,
                          hidden_size,
                          min_timescale=1.0,
                          max_timescale=1.0e4):
    """Return positional encoding.

  Calculates the position encoding as a mix of sine and cosine functions with
  geometrically increasing wavelengths.
  Defined and formulized in Attention is All You Need, section 3.5.

  Args:
    length: Sequence length.
    hidden_size: Size of the
    min_timescale: Minimum scale that will be applied at each position
    max_timescale: Maximum scale that will be applied at each position

  Returns:
    Tensor with shape [length, hidden_size]
  """
    position = tf.to_float(tf.range(length))#[0.0,1.0,2.0,...,length-1]
    num_timescales = hidden_size // 2
    log_timescale_increment = (
        math.log(float(max_timescale) / float(min_timescale)) /
        (tf.to_float(num_timescales) - 1))
    inv_timescales = min_timescale * tf.exp(
        tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(
        inv_timescales, 0)#[l,1]&[1,h/2]
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    return signal


def get_decoder_self_attention_bias(length):
    """Calculate bias for decoder that maintains model's autoregressive property.

  Creates a tensor that masks out locations that correspond to illegal
  connections, so prediction at position i cannot draw information from future
  positions.

  Args:
    length: int length of sequences in batch.

  Returns:
    float tensor of shape [1, 1, length, length]
  """
    with tf.name_scope("decoder_self_attention_bias"):
        #(L,L)下三角矩阵
        valid_locs = tf.matrix_band_part(tf.ones([length, length]), -1, 0)
        valid_locs = tf.reshape(valid_locs, [1, 1, length, length])#(1,1,L,L)
        #上三角矩阵，上半部分都是负无穷
        decoder_bias = _NEG_INF * (1.0 - valid_locs)
    return decoder_bias


def get_padding(inputs, inputs_unpadded_length):
    """Return float tensor representing the padding values in x.

  Args:
    inputs: float32 Tensor with shape  [batch_size, input_length, hidden_size]
    inputs_unpadded_length: int Tensor with shape [batch_size, 1]. The unpadded length of each input.

  Returns:
    float32 Tensor with shape [batch_size, input_length]. The padded locations will be 1 and unpadded with 0.
  """
    with tf.name_scope("padding"):
        #(32,75,256)
        input_shape = tf.shape(inputs)
        #[0,1,2,……,74]重复32次<-[0,1,2,……,74],[32]
        indexs = tf.tile(
            tf.range(input_shape[1]), tf.expand_dims(input_shape[0], axis=0))#张量扩充一定倍数
        #[[0,1,2,……,74],
        # [0,1,2,……,74],
        # ……
        # [0,1,2,……,74],
        # [0,1,2,……,74]](32,75)
        indexs = tf.reshape(indexs,
                            input_shape[:2])  # shape: [batch_size, input_length]
        #[[75,75,……,75],……,[75,75,……,75]](32,75)<-[[75],[75],……,[75]];[1,75]实际测试下来都是75
        inputs_unpadded_length = tf.tile(
            inputs_unpadded_length,
            tf.stack([1, input_shape[1]]))#[1,input_length]
        conditions = indexs < inputs_unpadded_length
        return 1 - tf.to_float(conditions) #给出一个[batch_size, input_length]矩阵，The padded locations will be 1 and unpadded with 0.填充的部分都是1

def get_padding_bias(inputs, inputs_unpadded_length):
    """Calculate bias tensor from padding values in tensor.

  Bias tensor that is added to the pre-softmax multi-headed attention logits,
  which has shape [batch_size, num_heads, length, length]. The tensor is zero at
  non-padding locations, and -1e9 (negative infinity) at padding locations.

  Args:
    x: int tensor with shape [batch_size, length]

  Returns:
    Attention bias tensor of shape [batch_size, 1, 1, length].
  """
    with tf.name_scope("attention_bias"):
        #(32,75)
        padding = get_padding(inputs, inputs_unpadded_length)
        attention_bias = padding * _NEG_INF#(32,75)都是负无穷
        #(32,1,1,75)
        attention_bias = tf.expand_dims(
            tf.expand_dims(attention_bias, axis=1), axis=1)
    return attention_bias