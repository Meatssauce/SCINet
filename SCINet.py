from typing import Tuple
import tensorflow as tf
from tensorflow.keras.regularizers import L1L2


class InnerConv1DBlock(tf.keras.layers.Layer):
    def __init__(self, filters: int, h: float, kernel_size: int, neg_slope: float = .01, dropout: float = .5, **kwargs):
        if filters <= 0 or h <= 0:
            raise ValueError('filters and h must be positive')
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


class SCIBlock(tf.keras.layers.Layer):
    def __init__(self, features: int, kernel_size: int, h: int, **kwargs):
        """
        :param features: number of features in the output
        :param kernel_size: kernel size of the convolutional layers
        :param h: scaling factor for convolutional module
        """
        super(SCIBlock, self).__init__(**kwargs)
        self.features = features
        self.kernel_size = kernel_size
        self.h = h

        self.conv1ds = {k: InnerConv1DBlock(filters=self.features, h=self.h, kernel_size=self.kernel_size, name=k)
                        for k in ['psi', 'phi', 'eta', 'rho']}  # regularize?

    def call(self, inputs, training=None):
        F_odd, F_even = inputs[:, ::2], inputs[:, 1::2]

        # Interactive learning as described in the paper
        F_s_odd = F_odd * tf.math.exp(self.conv1ds['phi'](F_even))
        F_s_even = F_even * tf.math.exp(self.conv1ds['psi'](F_odd))

        F_prime_odd = F_s_odd + self.conv1ds['rho'](F_s_even)
        F_prime_even = F_s_even - self.conv1ds['eta'](F_s_odd)

        return F_prime_odd, F_prime_even


class Interleave(tf.keras.layers.Layer):
    """A layer used to reverse the even-odd split operation."""

    def __init__(self, **kwargs):
        super(Interleave, self).__init__(**kwargs)

    def _interleave(self, slices):
        if not slices:
            return slices
        elif len(slices) == 1:
            return slices[0]

        mid = len(slices) // 2
        even = self._interleave(slices[:mid])
        odd = self._interleave(slices[mid:])

        shape = tf.shape(even)
        return tf.reshape(tf.stack([even, odd], axis=3), (shape[0], shape[1] * 2, shape[2]))

    def call(self, inputs):
        return self._interleave(inputs)


class SCINet(tf.keras.layers.Layer):
    def __init__(self, horizon: int, features: int, levels: int, h: int, kernel_size: int,
                 regularizer: Tuple[float, float] = (0, 0), **kwargs):
        """
        :param horizon: number of time stamps in output
        :param features: number of features in output
        :param levels: height of the binary tree + 1
        :param h: scaling factor for convolutional module in each SciBlock
        :param kernel_size: kernel size of convolutional module in each SciBlock
        :param regularizer: activity regularization (not implemented)
        """

        if levels < 1:
            raise ValueError('Must have at least 1 level')
        super(SCINet, self).__init__(**kwargs)
        self.horizon = horizon
        self.features = features
        self.levels = levels
        self.h = h
        self.kernel_size = kernel_size
        self.regularizer = regularizer

        self.interleave = Interleave()
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(
            horizon * features,
            kernel_regularizer=L1L2(0.001, 0.01),
            # activity_regularizer=L1L2(0.001, 0.01)
        )
        # self.regularizer = tf.keras.layers.ActivityRegularization(l1=regularizer[0], l2=regularizer[1])

        # tree of sciblocks
        self.sciblocks = [SCIBlock(features=features, kernel_size=kernel_size, h=h)
                          for _ in range(2 ** levels - 1)]

    def build(self, input_shape):
        if input_shape[1] / 2 ** self.levels % 1 != 0:
            raise ValueError(f'timestamps {input_shape[1]} must be evenly divisible by a tree with '
                             f'{self.levels} levels')
        super().build(input_shape)

    def call(self, inputs, training=None):
        # cascade input down a binary tree of sci-blocks
        lvl_inputs = [inputs]  # inputs for current level of the tree
        for i in range(self.levels):
            i_end = 2 ** (i + 1) - 1
            i_start = i_end - 2 ** i
            lvl_outputs = [output for j, tensor in zip(range(i_start, i_end), lvl_inputs)
                           for output in self.sciblocks[j](tensor)]
            lvl_inputs = lvl_outputs

        x = self.interleave(lvl_outputs)
        x += inputs

        # not sure if this is the correct way of doing it. The paper merely said to use a fully connected layer to
        # produce an output. Can't use TimeDistributed wrapper. It would force the layer's timestamps to match that of
        # the input -- something SCINet is supposed to solve
        x = self.flatten(x)
        x = self.dense(x)
        x = tf.reshape(x, (-1, self.horizon, self.features))

        return x

    def get_config(self):
        config = super().get_config()
        config.update({'horizon': self.horizon, 'features': self.features, 'levels': self.levels,
                       'kernel_size': self.kernel_size, 'h': self.h, 'regularizer': self.regularizer})
        return config


class StackedSCINet(tf.keras.layers.Layer):
    def __init__(self, horizon: int, features: int, stacks: int, levels: int, h: int, kernel_size: int,
                 regularizer: Tuple[float, float] = (0, 0), **kwargs):
        """
        :param horizon: number of time stamps in output
        :param stacks: number of stacked SciNets
        :param levels: number of levels for each SciNet
        :param h: scaling factor for convolutional module in each SciBlock
        :param kernel_size: kernel size of convolutional module in each SciBlock
        :param regularizer: activity regularization (not implemented)
        """
        if stacks < 1:
            raise ValueError('Must have at least 1 stack')
        super(StackedSCINet, self).__init__(**kwargs)

        self.horizon = horizon
        self.features = features
        self.levels = levels
        self.h = h
        self.kernel_size = kernel_size
        self.regularizer = regularizer

        self.scinets = [SCINet(horizon=horizon, features=features, levels=levels, h=h, kernel_size=kernel_size,
                               regularizer=regularizer) for _ in range(stacks)]
        self.mse_fn = tf.keras.metrics.MeanSquaredError()
        self.mae_fn = tf.keras.metrics.MeanAbsoluteError()

    def call(self, inputs, targets, sample_weights=None, training=None):
        outputs = []
        for scinet in self.scinets:
            x = scinet(inputs)
            outputs.append(x)  # keep each stack's output for intermediate supervision
            inputs = tf.concat([x, inputs[:, x.shape[1]:, :]], axis=1)  # X_hat_k concat X_(t-(T-tilda)+1:t)

        if targets is not None:
            # Calculate metrics
            mse = self.mse_fn(targets, x, sample_weights)
            mae = self.mae_fn(targets, x, sample_weights)
            self.add_metric(mse)
            self.add_metric(mae)

            # Calculate loss as sum of mean of norms of differences between output and input feature vectors for
            # each stack
            stacked_outputs = tf.stack(outputs)
            differences = stacked_outputs - targets
            loss = tf.linalg.normalize(differences, axis=3)[1]
            loss = tf.reduce_sum(loss, 2)
            loss /= self.horizon
            loss = tf.reduce_sum(loss)
            self.add_loss(loss)

        return x

    def get_config(self):
        config = super().get_config()
        config.update({'horizon': self.horizon, 'features': self.features, 'stacks': len(self.scinets),
                       'levels': self.levels, 'h': self.h, 'kernel_size': self.kernel_size,
                       'regularizer': self.regularizer})
        return config


class SCINetEndpoint(tf.keras.layers.Layer):
    def __init__(self, name=None):
        super(SCINetEndpoint, self).__init__(name=name)
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        self.accuracy_fn = tf.keras.metrics.MeanSquaredError()

    def call(self, targets, logits, sample_weights=None):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        loss = self.loss_fn(targets, logits, sample_weights)
        self.add_loss(loss)

        # Log accuracy as a metric and add it
        # to the layer using `self.add_metric()`.
        acc = self.accuracy_fn(targets, logits, sample_weights)
        self.add_metric(acc, name="mean_squared_error")

        # Return the inference-time prediction tensor (for `.predict()`).
        return logits
