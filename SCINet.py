from typing import Tuple
import os
import random
import pandas as pd
import numpy as np
import tensorflow as tf
from joblib import dump, load
from collections import namedtuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.regularizers import L1L2


class InnerConv1DBlock(tf.keras.layers.Layer):
    def __init__(self, filters: int, h: int, kernel_size: int, neg_slope: float = .01, dropout: float = .5,
                 name: str = ''):
        super(InnerConv1DBlock, self).__init__(name=name)
        self.conv1d = tf.keras.layers.Conv1D(h * filters, kernel_size, padding='same')
        self.leakyrelu = tf.keras.layers.LeakyReLU(neg_slope)

        self.dropout = tf.keras.layers.Dropout(dropout)

        self.conv1d2 = tf.keras.layers.Conv1D(filters, kernel_size, padding='same')
        self.tanh = tf.keras.activations.tanh

    def call(self, input_tensor):
        x = self.conv1d(input_tensor)
        x = self.leakyrelu(x)

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
    def __init__(self, kernel_size: int, h: int):
        super(SciBlock, self).__init__()
        self.kernel_size = kernel_size
        self.h = h

        self.split = Split()
        self.exp = Exp()

    def build(self, input_shape):
        _, _, filters = input_shape

        self.psi = InnerConv1DBlock(filters, self.h, self.kernel_size, name='psi')
        self.phi = InnerConv1DBlock(filters, self.h, self.kernel_size, name='phi')
        self.eta = InnerConv1DBlock(filters, self.h, self.kernel_size, name='eta')
        self.rho = InnerConv1DBlock(filters, self.h, self.kernel_size, name='rho')

    def call(self, input_tensor):
        F_odd, F_even = self.split(input_tensor)

        F_s_odd = F_odd * self.exp(self.phi(F_even))
        F_s_even = F_even * self.exp(self.psi(F_s_odd))

        F_prime_odd = F_s_odd + self.rho(F_s_even)
        F_prime_even = F_s_even - self.eta(F_s_odd)

        return F_prime_odd, F_prime_even


class Interleave(tf.keras.layers.Layer):
    def __init__(self):
        super(Interleave, self).__init__()

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


class SciNet(tf.keras.Model):
    def __init__(self, output_length: int, level: int, h: int, kernel_size: int):
        super(SciNet, self).__init__()
        self.level = level
        self.h = h
        self.kernel_size = kernel_size
        self.max_nodes = 2 ** (level + 1) - 1

        # self.sciblocks = [SciBlock(kernel_size, h) for _ in range(self.max_nodes)]
        self.interleave = Interleave()
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(output_length, kernel_regularizer=L1L2(0.001, 0.01))

        assert level == 3
        self.sciblock11 = SciBlock(kernel_size, h)
        self.sciblock21 = SciBlock(kernel_size, h)
        self.sciblock22 = SciBlock(kernel_size, h)
        self.sciblock31 = SciBlock(kernel_size, h)
        self.sciblock32 = SciBlock(kernel_size, h)
        self.sciblock33 = SciBlock(kernel_size, h)
        self.sciblock34 = SciBlock(kernel_size, h)

    def call(self, input_tensor):
        # cascade input down a binary tree of sci-blocks
        # inputs = [input_tensor]
        # for i in range(self.level):
        #     print('----')
        #     i_end = 2 ** (i + 1) - 1
        #     i_start = i_end - 2 ** i
        #     outputs = [out for tensor, j in zip(inputs, range(i_start, i_end)) for out in self.sciblocks[j](tensor)]
        #     inputs = outputs

        x11, x12 = self.sciblock11(input_tensor)

        x21, x22 = self.sciblock21(x11)
        x23, x24 = self.sciblock22(x12)

        x31, x32 = self.sciblock31(x21)
        x33, x34 = self.sciblock32(x22)
        x35, x36 = self.sciblock33(x23)
        x37, x38 = self.sciblock34(x24)

        # x = self.interleave(outputs)
        x = self.interleave([x31, x32, x33, x34, x35, x36, x37, x38])
        x += input_tensor

        x = self.flatten(x)
        x = self.dense(x)
        return x