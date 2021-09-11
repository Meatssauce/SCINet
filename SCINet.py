from typing import Tuple
import os
import random
import math
import pandas as pd
import numpy as np
import tensorflow as tf
from joblib import dump, load
from collections import namedtuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.regularizers import L1L2


class InnerConv1DBlock(tf.keras.layers.Layer):
    def __init__(self, filters: int, h: float, kernel_size: int, neg_slope: float = .01, dropout: float = .5,
                 name: str = '', **kwargs):
        assert filters > 0 and h > 0
        super(InnerConv1DBlock, self).__init__(name=name, **kwargs)
        self.conv1d = tf.keras.layers.Conv1D(max(round(h * filters), 1), kernel_size, padding='same')
        self.leakyrelu = tf.keras.layers.LeakyReLU(neg_slope)

        self.dropout = tf.keras.layers.Dropout(dropout)

        self.conv1d2 = tf.keras.layers.Conv1D(filters, kernel_size, padding='same')
        self.tanh = tf.keras.activations.tanh

    def call(self, input_tensor, training=None):
        x = self.conv1d(input_tensor)
        x = self.leakyrelu(x)

        if training:
            x = self.dropout(x)

        x = self.conv1d2(x)
        x = self.tanh(x)
        return x


class Exp(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Exp, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.math.exp(inputs)


class Split(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Split, self).__init__(**kwargs)

    def call(self, inputs):
        return inputs[:, ::2], inputs[:, 1::2]


class SciBlock(tf.keras.layers.Layer):
    def __init__(self, output_length: int, kernel_size: int, h: int, name: str = 'sci_block', **kwargs):
        super(SciBlock, self).__init__(name=name, **kwargs)
        self.h = h
        self.kernel_size = kernel_size
        # self.split = Split()
        # self.exp = Exp()

    def build(self, input_shape):
        self.conv1ds = {k: InnerConv1DBlock(input_shape[2], self.h, self.kernel_size, name=k)  # regularize?
                        for k in ['psi', 'phi', 'eta', 'rho']}
        # [layer.build(input_shape) for layer in self.conv1ds.values()]  # not needed?

    def call(self, inputs):
        F_odd, F_even = inputs[:, ::2], inputs[:, 1::2]

        F_s_odd = F_odd * tf.math.exp(self.conv1ds['phi'](F_even))
        F_s_even = F_even * tf.math.exp(self.conv1ds['psi'](F_s_odd))

        F_prime_odd = F_s_odd + self.conv1ds['rho'](F_s_even)
        F_prime_even = F_s_even - self.conv1ds['eta'](F_s_odd)

        return F_prime_odd, F_prime_even


class Interleave(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Interleave, self).__init__(**kwargs)

    def interleave(self, slices):
        if not slices:
            return slices
        elif len(slices) == 1:
            return slices[0]

        mid = len(slices) // 2
        even = self.interleave(slices[:mid])
        odd = self.interleave(slices[mid:])

        shape = tf.shape(even)
        return tf.reshape(tf.stack([even, odd], axis=3), (shape[0], shape[1]*2, shape[2]))

    def call(self, inputs):
        return self.interleave(inputs)


class SciNet(tf.keras.layers.Layer):
    def __init__(self, output_length: int, levels: int, h: int, kernel_size: int,
                 regularizer: Tuple[float, float] = (0, 0), name: str = '', **kwargs):
        super(SciNet, self).__init__(name=name, **kwargs)
        self.levels = levels
        self.interleave = Interleave()
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(output_length, kernel_regularizer=L1L2(0.001, 0.01))
        # self.regularizer = tf.keras.layers.ActivityRegularization(l1=regularizer[0], l2=regularizer[1])

        # tree of sciblocks
        self.sciblocks = [SciBlock(output_length, kernel_size, h) for _ in range(2 ** (levels + 1) - 1)]

    def build(self, input_shape):
        if input_shape[1] / 2 ** self.levels % 1 != 0:
            raise ValueError(f'timestamps {input_shape[1]} is not divisible by a tree with {self.levels} levels')
        [layer.build(input_shape) for layer in self.sciblocks]

    def call(self, inputs):
        # cascade input down a binary tree of sci-blocks
        lvl_inputs = [inputs]
        for i in range(self.levels):
            i_end = 2 ** (i + 1) - 1
            i_start = i_end - 2 ** i
            lvl_outputs = [out for j, tensor in zip(range(i_start, i_end), lvl_inputs)
                           for out in self.sciblocks[j](tensor)]
            lvl_inputs = lvl_outputs

        x = self.interleave(lvl_outputs)
        x += inputs

        x = self.flatten(x)
        x = self.dense(x)

        # x = self.regularizer(x)
        return x


# class StackedSciNet(tf.keras.layers.Layer):
#     def __init__(self, stacks: int, output_length: int, level: int, h: int, kernel_size: int,
#                  regularizer: Tuple[float, float] = (0, 0), name: str = '' **kwargs):
#         super(StackedSciNet, self).__init__(name=name, **kwargs)
#         assert stacks > 0
#         self.sci_nets = [SciNet(output_length, level, h, kernel_size, regularizer) for _ in range(stacks)]
#
#     def build(self, input_shape):
#         [stack.build(input_shape) for stack in self.stacks]
#
#     def call(self, inputs):
#         stack_outputs = []
#         for sci_net in self.sci_nets:
#             x = sci_net(x)
#             stack_outputs.append(x)
#
#         # calculate loss as sum of mean of norms of differences between output and input feature vectors for each stack
#         stack_outputs = tf.stack(stack_outputs)
#         loss = tf.linalg.normalize(stack_outputs - inputs, 2)[1]
#         loss = tf.reshape(loss, (-1, self.output_length))
#         loss = tf.reduce_sum(loss, 1)
#         loss = loss / self.output_length
#         loss = tf.reduce_sum(loss)
#         self.add_loss(loss)
#
#         return x
