import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import losses, layers
import tensorflow.experimental.numpy as tnp
from utils.utils import tf_tensor_list_slice

class HierarchicalContrastiveLoss(losses.Loss):
    def __init__(self, alpha=0.5, temporal_unit=0, **kwargs):
        """
        This module calculates hierarchical contrastive loss
        with instance contrastive loss and temporal contrastive loss
        :param alpha:
        :param temporal_unit:
        :param kwargs:
        """
        super().__init__()
        self.alpha = tf.cast(alpha, tf.float32)
        self.temporal_unit = temporal_unit
        self.instance_loss = InstanceContrastiveLoss()
        self.temporal_loss = TemporalContrastiveLoss()

    def call(self, z1, z2):
        loss = tf.constant(0, dtype=tf.float32)
        d = tf.constant(0, dtype=tf.float32)
        while tf.shape(z1)[1] > 1:
            if self.alpha != 0:
                loss += self.alpha * self.instance_loss(z1, z2)
            if d >= self.temporal_unit:
                loss += (1-self.alpha) * self.temporal_loss(z1, z2)
            d += 1
            z1 = layers.MaxPool1D(pool_size=2)(z1)
            z2 = layers.MaxPool1D(pool_size=2)(z2)
        if tf.shape(z1)[1] == 1:
            if self.alpha != 0:
                loss += self.alpha * self.instance_loss(z1, z2)
            d += 1
        return loss/d

class InstanceContrastiveLoss(losses.Loss):

    def __init__(self):
        """
        This module calculates instance contrastive loss
        """
        super(InstanceContrastiveLoss, self).__init__()

    def call(self, z1, z2):
        B, T = tf.shape(z1)[0], tf.shape(z1)[1]
        if B == 1:
            return tf.constant(0.)
        z = layers.concatenate([z1, z2], axis=0)  # 2B X T X C
        z = tf.transpose(z, perm=[1, 0, 2])  # T X 2B x C
        sim = tf.linalg.matmul(z, tf.transpose(z, perm=[0, 2, 1])) # T X 2B x 2B

        logits_tril = tf.linalg.set_diag(tf.linalg.band_part(sim, -1, 0), tf.zeros(tf.shape(sim)[0:-1]), name=None)
        logits_tril = tf.slice(
            logits_tril,
            [0, 0, 0], [tf.shape(logits_tril)[0], tf.shape(logits_tril)[1], tf.shape(logits_tril)[2] - 1]
        )
        logits_triu = tf.linalg.set_diag(tf.linalg.band_part(sim, 0, -1), tf.zeros(tf.shape(sim)[0:-1]), name=None)
        logits_triu = tf.slice(
            logits_triu,
            [0, 0, 1], [tf.shape(logits_triu)[0], tf.shape(logits_triu)[1], tf.shape(logits_triu)[2] - 1]
        )
        logits = logits_tril + logits_triu
        logits = -tf.nn.log_softmax(logits, axis=-1)

        i = tf.range(B)
        # return (tf.math.reduce_mean(tnp.asarray(logits)[:, i, B + i - 1]) + tf.math.reduce_mean(
        #     tnp.asarray(logits)[:, B + i, i])) / 2
        loss1 = tf.transpose(tf.gather_nd(tf.transpose(logits, [1, 2, 0]), tf.stack([i, B + i - 1], axis=1)))
        loss2 = tf.transpose(tf.gather_nd(tf.transpose(logits, [1, 2, 0]), tf.stack([B+i, i], axis=1)))
        return (tf.math.reduce_mean(loss1) + tf.math.reduce_mean(loss2)) / 2

class TemporalContrastiveLoss(losses.Loss):
    def __init__(self):
        """
        This module calculates temporal contrastive loss
        """
        super(TemporalContrastiveLoss, self).__init__()

    def call(self, z1, z2):
        B, T = tf.shape(z1)[0], tf.shape(z1)[1]
        if T == 1:
            return tf.constant(0.)
        z = layers.concatenate([z1, z2], axis=1)  # B X 2T X C
        sim = tf.linalg.matmul(z, tf.transpose(z, perm=[0, 2, 1])) # T X 2B x 2B

        logits_tril = tf.linalg.set_diag(tf.linalg.band_part(sim, -1, 0), tf.zeros(tf.shape(sim)[0:-1]), name=None)
        logits_tril = tf.slice(
            logits_tril,
            [0, 0, 0], [tf.shape(logits_tril)[0], tf.shape(logits_tril)[1], tf.shape(logits_tril)[2] - 1]
        )
        logits_triu = tf.linalg.set_diag(tf.linalg.band_part(sim, 0, -1), tf.zeros(tf.shape(sim)[0:-1]), name=None)
        logits_triu = tf.slice(
            logits_triu,
            [0, 0, 1], [tf.shape(logits_triu)[0], tf.shape(logits_triu)[1], tf.shape(logits_triu)[2] - 1]
        )
        logits = logits_tril + logits_triu
        logits = -tf.nn.log_softmax(logits, axis=-1)

        t = tf.range(T)
        # return (tf.math.reduce_mean(tnp.asarray(logits)[:, t, T + t - 1]) +
        #         tf.math.reduce_mean(tnp.asarray(logits)[:, T + t, t])) / 2
        loss1 = tf.transpose(tf.gather_nd(tf.transpose(logits, [1, 2, 0]), tf.stack([t, T + t - 1], axis=1)))
        loss2 = tf.transpose(tf.gather_nd(tf.transpose(logits, [1, 2, 0]), tf.stack([T + t, t], axis=1)))
        return (tf.math.reduce_mean(loss1) + tf.math.reduce_mean(loss2)) / 2

if __name__=="__main__":
    z1 = tnp.random.random((70, 3, 6))
    z2 = tnp.random.random((70, 3, 6))
    B, T = z1.shape[0], z1.shape[1]

    # INstance
    z = layers.concatenate([z1, z2], axis=0)  # 2B X T X C
    z = tf.transpose(z, perm=[1, 0, 2])  # T X 2B x C
    sim = tf.linalg.matmul(z, tf.transpose(z, perm=[0, 2, 1]))  # T X 2B x 2B
    logits_tril = tf.linalg.band_part(sim, -1, 0)
    logits_tril = tf.linalg.set_diag(logits_tril, tf.zeros(tf.shape(logits_tril)[0:-1]), name=None)
    logits_tril = tf.slice(logits_tril, [0, 0, 0],
                           [tf.shape(logits_tril)[0], tf.shape(logits_tril)[1], tf.shape(logits_tril)[2] - 1])
    logits_triu = tf.linalg.band_part(sim, 0, -1)
    logits_triu = tf.linalg.set_diag(logits_triu, tf.zeros(tf.shape(logits_triu)[0:-1]), name=None)
    logits_triu = tf.slice(logits_triu, [0, 0, 1],
                           [tf.shape(logits_triu)[0], tf.shape(logits_triu)[1], tf.shape(logits_triu)[2] - 1])
    logits = logits_tril + logits_triu
    logits = -tf.nn.log_softmax(logits, axis=-1)

    i = tf.range(B)
    tf.transpose(tf.gather_nd(tf.transpose(logits, [1, 2, 0]), tf.stack([i, B + i - 1], axis=1))) == \
    tnp.asarray(logits)[:, i, i + B - 1]

    tf.transpose(tf.gather_nd(tf.transpose(logits, [1, 2, 0]), tf.stack([B+i, i], axis=1))) == \
    tnp.asarray(logits)[:, B + i, i]
