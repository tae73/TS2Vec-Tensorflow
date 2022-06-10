import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.experimental.numpy as tnp
from utils.learn_utils import take_per_row
import tensorflow_probability as tfp

class RandomCropping(layers.Layer):
    def __init__(self, temporal_unit, input_shape):
        """
        This module generate two cropped time-series data from the 3-axis tensor
        """
        super(RandomCropping, self).__init__()
        self.temporal_unit = temporal_unit
        self.ts_l = input_shape[1]
        self.crop_l = tf.random.uniform(shape=[], minval=2**(self.temporal_unit+1), maxval=self.ts_l+1, dtype=tf.int32)
        self.crop_left = tf.random.uniform(shape=[], minval=0, maxval=self.ts_l-self.crop_l+1, dtype=tf.int32)
        self.crop_right = self.crop_left + self.crop_l
        self.crop_eleft = tf.random.uniform(shape=[], minval=0, maxval=self.crop_left+1, dtype=tf.int32)
        self.crop_eright = tf.random.uniform(shape=[], minval=self.crop_right, maxval=self.ts_l+1, dtype=tf.int32)
        self.crop_offset = tf.random.uniform(shape=[input_shape[0]], minval=-self.crop_eleft, maxval=self.ts_l-self.crop_eright+1, dtype=tf.int32)

    def call(self, inputs, **kwargs):
        return (take_per_row(inputs, self.crop_offset + self.crop_eleft, self.crop_right - self.crop_eleft),
                take_per_row(inputs, self.crop_offset + self.crop_left, self.crop_eright - self.crop_left)), \
               self.crop_l

class BinomialMask(layers.Layer):
    def __init__(self, p=0.5):
        """
        This module generate zero mask on time-series data from the binomial distribution
        :param p:
        """
        super(BinomialMask, self).__init__()
        self.dist = tfp.distributions.Binomial(total_count=1, probs=p)

    def call(self, B, T):
        return tf.cast(self.dist.sample(sample_shape=(B, T)), tf.bool)

class ContinuousMask(layers.Layer):
    def __init__(self, B, T, n=0.5, l=0.1):
        """
        This module generate zero continuous mask on time-series data
        :param B:
        :param T:
        :param n:
        :param l:
        """
        super(ContinuousMask, self).__init__()

        self.B = B
        self.T = T
        self.res = tf.Variable(tnp.full((B, T), True))
        if isinstance(n, float):
            n = int(n * T)
        self.n = tf.math.maximum(tf.math.minimum(n, T // 2), 1)

        if isinstance(l, float):
            l = int(l * T)
        self.l = tf.math.maximum(l, 1)

    def call(self):
        for i in range(self.B):
            for _ in range(self.n):
                t = tf.random.uniform(minval=0, maxval=self.T-self.l+1)
                self.res[i, t:t + l].assign(tf.constant(shape=(l), value=False))
        return self.res

def create_nan_mask(inputs):
    nan_mask = tf.math.reduce_any(tf.math.is_nan(inputs), axis=-1)
    nan_mask = ~tf.expand_dims(nan_mask, axis=len(nan_mask.shape))
    nan_mask_inputs = tf.where(~nan_mask, tf.zeros_like(inputs), inputs)
    return nan_mask_inputs, nan_mask

def create_final_mask(ts_mask, nan_mask, input_projection): # TODO 맞는지 확인
    ts_mask = tf.expand_dims(ts_mask, axis=len(ts_mask.shape))
    mask = tf.math.logical_and(ts_mask, nan_mask)
    mask_output = tf.where(~mask, tf.zeros_like(input_projection), input_projection) #True에 0
    return mask_output

# def generate_continuous_mask(B, T, n=5, l=0.1):
#     res = tf.Variable(tnp.full((B, T), True))
#     if isinstance(n, float):
#         n = int(n * T)
#     n = max(min(n, T // 2), 1)
#
#     if isinstance(l, float):
#         l = int(l * T)
#     l = max(l, 1)
#
#     for i in range(B):
#         for _ in range(n):
#             t = tnp.random.randint(T - l + 1)
#             res[i, t:t + l].assign(tf.constant(shape=(l), value=False))
#     return tf.constant(res)  # TODO constant로 변경?
#
#
