from typing import Tuple
import tensorflow as tf
from tensorflow.keras.regularizers import L1L2


class InnerConv1DBlock(tf.keras.layers.Layer):
    def __init__(self, filters: int, h: float, kernel_size: int, neg_slope: float = .01, dropout: float = .5, **kwargs):
        assert filters > 0 and h > 0
        super(InnerConv1DBlock, self).__init__(**kwargs)
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


# class Exp(tf.keras.layers.Layer):
#     def __init__(self, **kwargs):
#         super(Exp, self).__init__(**kwargs)
#
#     def call(self, inputs):
#         return tf.math.exp(inputs)
#
#
# class Split(tf.keras.layers.Layer):
#     def __init__(self, **kwargs):
#         super(Split, self).__init__(**kwargs)
#
#     def call(self, inputs):
#         return inputs[:, ::2], inputs[:, 1::2]


class SciBlock(tf.keras.layers.Layer):
    def __init__(self, output_length: int, kernel_size: int, h: int, **kwargs):
        super(SciBlock, self).__init__(**kwargs)
        self.h = h
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv1ds = {k: InnerConv1DBlock(input_shape[2], self.h, self.kernel_size, name=k)  # regularize?
                        for k in ['psi', 'phi', 'eta', 'rho']}
        # [layer.build(input_shape) for layer in self.conv1ds.values()]  # unneeded?

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
                 regularizer: Tuple[float, float] = (0, 0), **kwargs):
        super(SciNet, self).__init__(**kwargs)
        self.levels = levels
        self.interleave = Interleave()
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(output_length, kernel_regularizer=L1L2(0.001, 0.01))
        # self.regularizer = tf.keras.layers.ActivityRegularization(l1=regularizer[0], l2=regularizer[1])

        # tree of sciblocks
        self.sciblocks = [SciBlock(output_length, kernel_size, h) for _ in range(2 ** (levels + 1) - 1)]

    def build(self, input_shape):
        if input_shape[1] / 2 ** self.levels % 1 != 0:
            raise ValueError(f'timestamps {input_shape[1]} must be evenly divisible by a tree with '
                             f'{self.levels} levels')
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


class StackedSciNet(tf.keras.layers.Layer):
    def __init__(self, output_length: int, stacks: int, levels: int, h: int, kernel_size: int,
                 regularizer: Tuple[float, float] = (0, 0), **kwargs):
        if stacks < 1:
            raise ValueError('Must have at least 1 stack')
        super(StackedSciNet, self).__init__(**kwargs)
        self.output_length = output_length
        self.scinets = [SciNet(output_length, levels, h, kernel_size, regularizer) for _ in range(stacks)]
        self.mse_fn = tf.keras.metrics.MeanSquaredError()
        self.mae_fn = tf.keras.metrics.MeanAbsoluteError()

    def build(self, input_shape):
        [stack.build(input_shape) for stack in self.scinets]

    def call(self, inputs, targets=None, sample_weights=None):
        x = inputs
        outputs = []
        for scinet in self.scinets:
            x = scinet(x)
            outputs.append(x)  # keep each stack's output for intermediate supervision

        if targets is not None:
            # Calculate loss as sum of mean of norms of differences between output and input feature vectors for
            # each stack
            outputs = tf.stack(outputs)
            temp = outputs - targets
            loss = tf.linalg.normalize(temp, axis=1)[1]
            loss = tf.reshape(loss, (-1, self.output_length))
            loss = tf.reduce_sum(loss, 1)
            loss = loss / self.output_length
            loss = tf.reduce_sum(loss)
            self.add_loss(loss)

            # Calculate metrics
            mse = self.mse_fn(targets, x, sample_weights)
            mae = self.mae_fn(targets, x, sample_weights)
            self.add_metric(mse, name='mean_squared_error')
            self.add_metric(mse, name='mean_absolute_error')

        return x
