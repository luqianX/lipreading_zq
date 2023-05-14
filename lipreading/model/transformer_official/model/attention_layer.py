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
"""Implementation of multiheaded attention and self-attention layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

#Attention也不过是普通的一层
class Attention(tf.layers.Layer):
    """Multi-headed attention layer."""

    def __init__(self, hidden_size, num_heads, attention_dropout, train):
        #其中hidden_size即dmodel
        if hidden_size % num_heads != 0:
            raise ValueError(
                "Hidden size must be evenly divisible by the number of "
                "heads.")

        super(Attention, self).__init__()
        self.hidden_size = hidden_size#
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.train = train

        # Layers for linearly projecting the queries, keys, and values.
        #(32,75,256)<-(32,75,256)
        self.q_dense_layer = tf.layers.Dense(
            hidden_size, use_bias=False, name="q")#线性计算，hidden_size是输出大小
        self.k_dense_layer = tf.layers.Dense(
            hidden_size, use_bias=False, name="k")
        self.v_dense_layer = tf.layers.Dense(
            hidden_size, use_bias=False, name="v")
        
        self.output_dense_layer = tf.layers.Dense(
            hidden_size, use_bias=False, name="output_transform")
        # self.output_dense_layer = tf.layers.Conv1D(
        # hidden_size,
        # kernel_size=1,
        # use_bias=False,
        # name="output_transform")

    def split_heads(self, x):
        """Split x into different heads, and transpose the resulting value.

    The tensor is transposed to insure the inner dimensions hold the correct
    values during the matrix multiplication.

    Args:
      x: A tensor with shape [batch_size, length, hidden_size]

    Returns:
      A tensor with shape [batch_size, num_heads, length, hidden_size/num_heads]
    """
        with tf.name_scope("split_heads"):
            #(32,75,256)
            batch_size = tf.shape(x)[0]
            length = tf.shape(x)[1]

            # Calculate depth of last dimension after it has been split.32
            depth = (self.hidden_size // self.num_heads)

            # Split the last dimension(32,75,8,32)
            x = tf.reshape(x, [batch_size, length, self.num_heads, depth])

            # Transpose the result(32,8,75,32)
            return tf.transpose(x, [0, 2, 1, 3])

    def combine_heads(self, x):
        """Combine tensor that has been split.

    Args:
      x: A tensor [batch_size, num_heads, length, hidden_size/num_heads]

    Returns:
      A tensor with shape [batch_size, length, hidden_size]
    """
        with tf.name_scope("combine_heads"):
            #(32,8,75,32)
            batch_size = tf.shape(x)[0]#32
            length = tf.shape(x)[2]#75
            #(32,75,8,32)
            x = tf.transpose(
                x, [0, 2, 1, 3])  # --> [batch, length, num_heads, depth]
            #(32,75,256)
            return tf.reshape(x, [batch_size, length, self.hidden_size])

    def call(self, x, y, bias, cache=None):
        #(32,75,256)<-(32,75,256)
        #(32,L,256)<-(32,L,256)
        """Apply attention mechanism to x and y.

    Args:
      x: a tensor with shape [batch_size, length_x, hidden_size](32,75,256)
      y: a tensor with shape [batch_size, length_y, hidden_size]
      bias: attention bias that will be added to the result of the dot product.
      cache: (Used during prediction) dictionary with tensors containing results
        of previous attentions. The dictionary must have the items:
            {"k": tensor with shape [batch_size, i, key_channels],
             "v": tensor with shape [batch_size, i, value_channels]}
        where i is the current decoded length.

    Returns:
      Attention layer output with shape [batch_size, length_x, hidden_size]
    """
        # Linearly project the query (q), key (k) and value (v) using different
        # learned projections. This is in preparation of splitting them into
        # multiple heads. Multi-head attention uses multiple queries, keys, and
        # values rather than regular attention (which uses a single q, k, v).
        #(32,75,256)
        #(32,L,256)
        q = self.q_dense_layer(x)
        k = self.k_dense_layer(y)
        v = self.v_dense_layer(y)

        if cache is not None:
            # Combine cached keys and values with new keys and values.
            k = tf.concat([cache["k"], k], axis=1)
            v = tf.concat([cache["v"], v], axis=1)

            # Update cache
            cache["k"] = k
            cache["v"] = v

        # Split q, k, v into heads.(32,8,75,32)
        # Split q, k, v into heads.(32,8,L,32)
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # Scale q to prevent the dot product between q and k from growing too large.
        depth = (self.hidden_size // self.num_heads)#32
        q *= depth**-0.5#除以根号dk

        # Calculate dot product attention
        #(32,8,75,75)<-(32,8,75,32),(32,8,75,32)
        #(32,8,L,L)<-(32,8,L,32),(32,8,L,32)
        logits = tf.matmul(q, k, transpose_b=True)#注意第二个参数
        logits += bias#就是依据这个bias矩阵使得解码器保持自回归特性
        weights = tf.nn.softmax(logits, name="attention_weights")
        if self.train:
            weights = tf.nn.dropout(weights, 1.0 - self.attention_dropout)
        #(32,8,75,32)<-(32,8,75,75),(32,8,75,32)
        #(32,8,L,32)<-(32,8,L,L),(32,8,L,32)
        attention_output = tf.matmul(weights, v)

        # Recombine heads --> [batch_size, length, hidden_size]
        #(32,75,256)
        #(32,L,256)
        attention_output = self.combine_heads(attention_output)

        # Run the combined outputs through another linear projection layer.
        # (32,75,256)<-(32,75,256)
        # (32,L,256) <-(32,L,256)
        attention_output = self.output_dense_layer(attention_output)
        return attention_output


class SelfAttention(Attention):
    """Multiheaded self-attention layer."""

    def call(self, x, bias, cache=None):
        return super(SelfAttention, self).call(x, x, bias, cache)
