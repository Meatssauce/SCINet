import tensorflow as tf


class InnerConv1DBlock(tf.keras.layers.Layer):
    def __init__(self, filters: int, h: float, kernel_size: int, neg_slope: float = .01, dropout: float = .5,
                 **kwargs):
        if filters <= 0 or h <= 0:
            raise ValueError('filters and h must be positive')

        super().__init__(**kwargs)
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
    def __init__(self, features: int, kernel_size: int, h: int, name='sciblock', **kwargs):
        """
        :param features: number of features in the output
        :param kernel_size: kernel size of the convolutional layers
        :param h: scaling factor for convolutional module
        """
        super().__init__(name=name, **kwargs)
        self.features = features
        self.kernel_size = kernel_size
        self.h = h

        self.conv1ds = {k: InnerConv1DBlock(filters=self.features, h=self.h, kernel_size=self.kernel_size, name=k)
                        for k in ['psi', 'phi', 'eta', 'rho']}  # regularize?

    def call(self, inputs):
        F_odd, F_even = inputs[:, ::2], inputs[:, 1::2]

        # Interactive learning as described in the paper
        F_s_odd = F_odd * tf.math.exp(self.conv1ds['phi'](F_even))
        F_s_even = F_even * tf.math.exp(self.conv1ds['psi'](F_odd))

        F_prime_odd = F_s_odd + self.conv1ds['rho'](F_s_even)
        F_prime_even = F_s_even - self.conv1ds['eta'](F_s_odd)

        return F_prime_odd, F_prime_even

    def get_config(self):
        config = super().get_config()
        config.update({'features': self.features, 'kernel_size': self.kernel_size, 'h': self.h})
        return config


class Interleave(tf.keras.layers.Layer):
    """A layer used to reverse the even-odd split operation."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
                 kernel_regularizer=None, activity_regularizer=None, name='scinet', **kwargs):
        """
        :param horizon: number of time stamps in output
        :param levels: height of the binary tree + 1
        :param h: scaling factor for convolutional module in each SCIBlock
        :param kernel_size: kernel size of convolutional module in each SCIBlock
        :param kernel_regularizer: kernel regularizer for the fully connected layer at the end
        :param activity_regularizer: activity regularizer for the fully connected layer at the end
        """
        if levels < 1:
            raise ValueError('Must have at least 1 level')

        super().__init__(name=name, **kwargs)
        self.horizon = horizon
        self.features = features
        self.levels = levels
        self.h = h
        self.kernel_size = kernel_size

        self.interleave = Interleave()
        self.flatten = tf.keras.layers.Flatten()

        # tree of sciblocks
        self.sciblocks = [SCIBlock(features=features, kernel_size=self.kernel_size, h=self.h)
                          for _ in range(2 ** self.levels - 1)]
        self.dense = tf.keras.layers.Dense(
            self.horizon * features,
            kernel_regularizer=kernel_regularizer,
            activity_regularizer=activity_regularizer
        )

    def build(self, input_shape):
        if input_shape[1] / 2 ** self.levels % 1 != 0:
            raise ValueError(f'timestamps {input_shape[1]} must be evenly divisible by a tree with '
                             f'{self.levels} levels')
        super().build(input_shape)

    def call(self, inputs):
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
        config.update({'horizon': self.horizon, 'levels': self.levels})
        return config


class StackedSCINet(tf.keras.layers.Layer):
    """Layer that implements StackedSCINet as described in the paper.

    When called, outputs a tensor of shape (K, -1, n_steps, n_features) containing the outputs of all K internal
    SCINets (e.g., output[k-1] is the output of the kth SCINet, where k is in [1, ..., K]).

    To use intermediate supervision, pass the layer's output to StackedSCINetLoss as a separate model output.
    """

    def __init__(self, horizon: int, features: int, stacks: int, levels: int, h: int, kernel_size: int,
                 kernel_regularizer=None, activity_regularizer=None, name='stacked_scinet', **kwargs):
        """
        :param horizon: number of time stamps in output
        :param stacks: number of stacked SCINets
        :param levels: number of levels for each SCINet
        :param h: scaling factor for convolutional module in each SCIBlock
        :param kernel_size: kernel size of convolutional module in each SCIBlock
        :param kernel_regularizer: kernel regularizer for each SCINet
        :param activity_regularizer: activity regularizer for each SCINet
        """
        if stacks < 2:
            raise ValueError('Must have at least 2 stacks')

        super().__init__(name=name, **kwargs)
        self.stacks = stacks
        self.scinets = [SCINet(horizon=horizon, features=features, levels=levels, h=h,
                               kernel_size=kernel_size, kernel_regularizer=kernel_regularizer,
                               activity_regularizer=activity_regularizer) for _ in range(stacks)]

    def call(self, inputs):  # sample_weights=None
        outputs = []
        for scinet in self.scinets:
            x = scinet(inputs)
            outputs.append(x)  # keep each stack's output for intermediate supervision
            inputs = tf.concat([x, inputs[:, x.shape[1]:, :]], axis=1)  # X_hat_k concat X_(t-(T-tilda)+1:t)
        return tf.stack(outputs)

    def get_config(self):
        config = super().get_config()
        config.update({'stacks': self.stacks})
        return config


class Identity(tf.keras.layers.Layer):
    """Identity layer used solely for the purpose of naming model outputs and properly displaying outputs when plotting
    some multi-output models.

    Returns input without changing them.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        return tf.identity(inputs)


class StackedSCINetLoss(tf.keras.losses.Loss):
    """Compute loss for a Stacked SCINet via intermediate supervision.

    `loss = sum of mean normalised difference between each stack's output and ground truth`

    `y_pred` should be the output of a StackedSCINet layer.
    """

    def __init__(self, name='stacked_scienet_loss', **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        stacked_outputs = y_pred
        horizon = stacked_outputs.shape[2]

        errors = stacked_outputs - y_true
        loss = tf.linalg.normalize(errors, axis=3)[1]
        loss = tf.reduce_sum(loss, 2)
        loss /= horizon
        loss = tf.reduce_sum(loss)

        return loss


# class NetConcatenate(tf.keras.layer.Layer):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.concatenate = tf.keras.layers.Concatenate(axis=1)
#
#     def call(self, intermediates, inputs):
#         return self.concatenate([intermediates, inputs[:, intermediates.shape[1]:, :]])


def make_simple_scinet(input_shape, horizon: int, L: int, h: int, kernel_size: int, learning_rate: float,
                       kernel_regularizer=None, activity_regularizer=None, diagram_path=None):
    """Compiles a simple SCINet and saves model diagram if given a path.

    Intended to be a demonstration of simple model construction. See paper for details on the hyperparameters.
    """
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(input_shape[1], input_shape[2]), name='inputs'),
        SCINet(horizon, features=input_shape[-1], levels=L, h=h, kernel_size=kernel_size,
               kernel_regularizer=kernel_regularizer, activity_regularizer=activity_regularizer)
    ])

    model.summary()
    if diagram_path:
        tf.keras.utils.plot_model(model, to_file=diagram_path, show_shapes=True)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mse',
                  metrics=['mse', 'mae']
                  )

    return model


def make_simple_stacked_scinet(input_shape, horizon: int, K: int, L: int, h: int, kernel_size: int,
                               learning_rate: float, kernel_regularizer=None, activity_regularizer=None,
                               diagram_path=None):
    """Compiles a simple StackedSCINet and saves model diagram if given a path.

    Intended to be a demonstration of simple model construction. See paper for details on the hyperparameters.
    """
    inputs = tf.keras.Input(shape=(input_shape[1], input_shape[2]), name='lookback_window')
    x = StackedSCINet(horizon=horizon, features=input_shape[-1], stacks=K, levels=L, h=h,
                      kernel_size=kernel_size, kernel_regularizer=kernel_regularizer,
                      activity_regularizer=activity_regularizer)(inputs)
    outputs = Identity(name='outputs')(x[-1])
    intermediates = Identity(name='intermediates')(x)
    model = tf.keras.Model(inputs=inputs, outputs=[outputs, intermediates])

    model.summary()
    if diagram_path:
        tf.keras.utils.plot_model(model, to_file=diagram_path, show_shapes=True)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss={
                      # 'outputs': 'mse',
                      'intermediates': StackedSCINetLoss()
                  },
                  metrics={'outputs': ['mse', 'mae']}
                  )

    return model
