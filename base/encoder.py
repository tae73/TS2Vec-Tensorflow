from typing import List
import tensorflow.experimental.numpy as tnp
import tensorflow as tf
from tensorflow.keras import layers, models, initializers
from base.augmentation import create_nan_mask, create_final_mask, BinomialMask


class TimeSeriesEncoder(models.Model):
    def __init__(self,
                 depth: int=10, hidden_size: int=64, out_size: int=320,  projection_activation: str=None, kernel_size: int=3,
                 extractor_padding: str='causal', extractor_activation: str='gelu', extractor_dropout_rate: float = 0,
                 extractor_batch_norm: bool=True, extractor_layer_norm: bool = False,
                 represent_dropout_rate: float=0.1, mask_mode: str='binomial', random_seed: int=None
                 ):
        """
        This module creates TS2Vec network and manages learning procedure

        :param depth:
        :param hidden_size:
        :param out_size:
        :param projection_activation:
        :param kernel_size:
        :param extractor_padding:
        :param extractor_activation:
        :param extractor_dropout_rate:
        :param extractor_batch_norm:
        :param extractor_layer_norm:
        :param represent_dropout_rate:
        :param mask_mode:
        :param random_seed:
        """
        super(TimeSeriesEncoder, self).__init__()

        self.mask_mode = mask_mode
        self.binormial_mask = BinomialMask(p=0.5)
        # input projection layer
        self.projection_layer = layers.Dense(
            units=hidden_size, kernel_initializer=initializers.he_normal(seed=random_seed),
            bias_initializer='zeros', activation=projection_activation
        )
        self._create_extractors(
            channels=[hidden_size] * depth + [out_size], kernel_size=kernel_size,
            padding=extractor_padding, activation=extractor_activation, dropout=extractor_dropout_rate,
            random_seed=random_seed, batch_norm=extractor_batch_norm, layer_norm=extractor_layer_norm
            )

        self.represent_dropout = layers.SpatialDropout1D(rate=represent_dropout_rate)

        
    def _create_extractors(self, channels, kernel_size, padding, activation, dropout, random_seed, batch_norm, layer_norm):
        """

        :return: The encoder contains "levels" temporal dilated conv block
        """
        self.extractor_lsts = []
        for level, filters in enumerate(channels):
            dilation_rate = 2 ** level  # exponential growth
            self.extractor_lsts.append(
                DilatedResidualBlock(
                    filters=filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding=padding,
                    groups=1, activation=activation, dropout=dropout, random_seed=random_seed,
                    batch_norm=batch_norm, layer_norm=layer_norm
                )
            )

    def call(self, inputs, mask=None, training=None):
        inputs, nan_mask = create_nan_mask(inputs)  #TODO create_nan_mask 같은지 확

        # input projection
        input_projection = self.projection_layer(inputs)  # B x T x Ch

        # generate & apply mask
        if mask is None:
            if training:
                mask = self.mask_mode
            else:
                mask = 'all_true'

        if mask == 'binomial':
            ts_mask = self.binormial_mask(tf.shape(input_projection)[0], tf.shape(input_projection)[1])

        elif mask == 'all_true':
            ts_mask = tnp.full((tf.shape(input_projection)[0], tf.shape(input_projection)[1]), True)

        masked_projection = create_final_mask(ts_mask, nan_mask, input_projection)  # TODO 맞는지 확인

        represents = masked_projection
        for extractor in self.extractor_lsts:
            represents = extractor(represents, training=training)

        if training:
            return self.represent_dropout(represents, training=training)
        else:
            return represents

class DilatedResidualBlock(layers.Layer):
    def __init__(self,
                 filters: int, kernel_size: int, dilation_rate: int, padding: str, groups: int = 1,
                 activation: str = 'gelu', dropout: float = 0.0,  random_seed=None, batch_norm: bool = True, layer_norm: bool = False,
                 **kwargs
                 ):
        """
        1D Convolution block to handle time-series data with dilated convolution and residual connection
        note that, final output does not contains activation function
        :param filters:
        :param kernel_size:
        :param dilation_rate:
        :param padding:
        :param groups:
        :param activation:
        :param dropout:
        :param spatial_dropout:
        :param random_seed:
        :param batch_norm:
        :param layer_norm:
        :param kwargs:
        """
        super(DilatedResidualBlock, self).__init__()

        if batch_norm + layer_norm > 1:
            raise ValueError('Only one normalization can be specified at once.')

        if padding != 'causal' and padding != 'same':
            raise ValueError("Only 'causal' or 'same' padding are compatible.")

        # conv 1
        self.conv_layer_1 = layers.Conv1D(
            filters=filters, kernel_size=kernel_size, dilation_rate=dilation_rate, groups=groups,
            padding=padding, kernel_initializer=initializers.he_normal(seed=random_seed)
        )
        if batch_norm:
            self.normalization_layer_1 = layers.BatchNormalization(axis=-1)
        elif layer_norm:
            self.normalization_layer_1 = layers.LayerNormalization(axis=-1)
        else:
            self.normalization_layer_1 = None
        # TODO ADD weight normalization (tensorflow_addons)
        self.activation_layer_1 = layers.Activation(activation)
        self.dropout_layer_1 = layers.SpatialDropout1D(rate=dropout)
        # conv 2
        self.conv_layer_2 = layers.Conv1D(
            filters=filters, kernel_size=kernel_size, dilation_rate=dilation_rate, groups=groups,
            padding=padding, kernel_initializer=initializers.he_normal(seed=random_seed)
        )
        if batch_norm:
            self.normalization_layer_2 = layers.BatchNormalization(axis=-1)
        elif layer_norm:
            self.normalization_layer_2 = layers.LayerNormalization(axis=-1)
        else :
            self.normalization_layer_2 = None
        # TODO ADD weight normalization (tensorflow_addons)
        self.activation_layer_2 = layers.Activation(activation)
        self.dropout_layer_2 = layers.SpatialDropout1D(rate=dropout)

        self.downsample_layer = layers.Conv1D(
            filters=filters, kernel_size=1, padding='same', kernel_initializer=initializers.he_normal(seed=random_seed)
        )
        # self.output_layer = layers.Activation(activation)

    def call(self, inputs, training):
        residual = inputs
        # conv 1
        conv1 = self.conv_layer_1(inputs)
        if self.normalization_layer_1:
            if training: conv1 = self.normalization_layer_1(conv1)
        conv1 = self.activation_layer_1(conv1)
        if training: conv1 = self.dropout_layer_1(conv1) if training else conv1
        # conv2
        conv2 = self.conv_layer_2(conv1)
        if self.normalization_layer_2:
            if training: conv2 = self.normalization_layer_2(conv2)
        conv2 = self.activation_layer_2(conv2)
        if training: conv2 = self.dropout_layer_2(conv2)

        # if tf.shape(residual)[-1] != tf.shape(conv2)[-1]:
        if residual.shape[-1] != conv2.shape[-1]:
            residual = self.downsample_layer(residual)
        # assert residual.shape == conv2.shape

        # return self.output_layer(residual + conv2)
        return residual+conv2