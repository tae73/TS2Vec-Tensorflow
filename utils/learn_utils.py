import numpy as np
import tensorflow.experimental.numpy as tnp
import tensorflow as tf
import math

class CosineScheduler(object):
    def __init__(self, max_update, base_lr=0.01, final_lr=0,
               warmup_steps=0, warmup_begin_lr=0):
        self.base_lr_orig = base_lr
        self.max_update = max_update
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = self.max_update - self.warmup_steps

    def get_warmup_lr(self, epoch):
        increase = (self.base_lr_orig - self.warmup_begin_lr) \
                       * float(epoch) / float(self.warmup_steps)
        return self.warmup_begin_lr + increase

    def __call__(self, epoch):
        if epoch < self.warmup_steps:
            return self.get_warmup_lr(epoch)
        if epoch <= self.max_update:
            self.base_lr = self.final_lr + (
                self.base_lr_orig - self.final_lr) * (1 + math.cos(
                math.pi * (epoch - self.warmup_steps) / self.max_steps)) / 2
        return self.base_lr

def pad_nan(array, left=0, right=0, axis=0):
    if left > 0:
        pad_shape = list(array.shape)
        pad_shape[axis] = left
        array = tnp.concatenate((tnp.full(pad_shape, np.nan), array), axis=axis)
    if right > 0:
        pad_shape = list(array.shape)
        pad_shape[axis] = left
        array = tnp.concatenate(array, (tnp.full(pad_shape, np.nan)), axis=axis)
    return array

def pad_nan_to_target(array, target_length, axis=0, both_side=False):
    """

    :param array:
    :param target_length:
    :param axis: axis=0 then batch pad, axis=1 then timestep pad, axis=2 then feature pad
    :param both_side:
    :return:
    """
    # assert array.dtype in [np.float16, np.float32, np.float64]
    pad_size = target_length - array.shape[axis]
    if pad_size <= 0:
        return array
    npad = [(0, 0)] * array.ndim
    if both_side:
        npad[axis] = (pad_size // 2, pad_size - pad_size//2)
    else:
        npad[axis] = (0, pad_size)
    return np.pad(array, pad_width=npad, mode='constant', constant_values=np.nan)

def split_with_nan(inputs, selections, axis=0):
    # assert array.dtype in [np.float16, np.float32, np.float64]
    arrs = np.array_split(inputs, selections, axis=axis) # TODO 계산 중간에 array_split 상관 없나..?
    target_length = arrs[0].shape[axis]
    for i in range(len(arrs)):
        arrs[i] = pad_nan_to_target(arrs[i], target_length, axis=axis)
    return arrs

def centerize_vary_length_series(inputs):
    prefix_zeros = np.argmax(~np.isnan(inputs).all(axis=-1), axis=1)
    suffix_zeros = np.argmax(~np.isnan(inputs[:, ::-1]).all(axis=-1), axis=1)
    offset = (prefix_zeros + suffix_zeros) // 2 - prefix_zeros
    rows, column_indices = np.ogrid[:inputs.shape[0], :inputs.shape[1]]
    offset[offset < 0] += inputs.shape[1]
    column_indices = column_indices - offset[:, np.newaxis]
    return inputs[rows, column_indices]

def take_per_row(A, indx, num_elem):
    # all_indx = indx[:, None] + tnp.arange(num_elem)
    # return tnp.asarray(A)[tnp.arange(all_indx.shape[0])[:,None], all_indx]
    all_indx = tf.expand_dims(indx, axis=-1) + tf.range(num_elem)
    return tf.gather(params=A, indices=all_indx, batch_dims=1) # TODO 맞는지 확인


if __name__ == "__main__":
    scheduler = CosineScheduler(20, warmup_steps=5, base_lr=0.3, final_lr=0.01)
    scheduler(22)

