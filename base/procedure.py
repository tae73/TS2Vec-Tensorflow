from typing import Dict
from pathlib import Path
import time
import numpy as np
from tqdm import tqdm
import mlflow
import hydra
import tensorflow as tf
from tensorflow.keras import layers, models, losses, optimizers
import tensorflow.experimental.numpy as tnp
from base.loss import HierarchicalContrastiveLoss
from base.augmentation import RandomCropping
from utils.learn_utils import split_with_nan, centerize_vary_length_series, pad_nan
from utils.utils import log_params_from_omegaconf_dict


class NetworkLearningProcedure(object):
    def __init__(self, model, config, distribute_strategy):
        self.model = None
        self.tape = None

        self.result_state = None

    def learn(self):
        raise NotImplementedError

    def create_train_dataset(self):
        raise NotImplementedError

    def create_valid_dataset(self):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError

    def train_one_step(self):
        raise NotImplementedError

    def valid_one_step(self):
        raise NotImplementedError

    def distributed_train_step(self):
        raise NotImplementedError

    def distributed_valid_step(self):
        raise NotImplementedError

    def train_one_epoch(self):
        raise NotImplementedError

    def valid_one_epoch(self):
        raise NotImplementedError

class HierarchicalContrastiveLearn(NetworkLearningProcedure):
    def __init__(self, model, config, distribute_strategy):
        """
        This class is a module for learning time-series representation.
        :param model: TS2Vec Tensorflow model
        :param config: DictConfig
        :param distribute_strategy: tensorflow distributed strategy
        """
        super(HierarchicalContrastiveLearn, self).__init__(model, config, distribute_strategy)
        self.model: models.Model = model
        self.tape = None
        self.strategy = distribute_strategy

        self.config = config
        self.experiment_name = config.mlflow.experiment_name

        self.epochs: int = eval(config.ts2vec.train.epochs)
        self.iters: int = eval(config.ts2vec.train.iters)
        self.learning_rate: float = config.ts2vec.train.learning_rate
        self.max_train_length: int = eval(config.ts2vec.train.max_train_length)
        self.batch_size: int = config.ts2vec.train.batch_size
        self.buffer_size: int = config.ts2vec.train.buffer_size

        self.alpha: float = config.ts2vec.loss.alpha
        self.temporal_unit: float = config.ts2vec.loss.temporal_unit
        self.random_seed: int = eval(config.random.random_seed)

        if self.strategy is not None:
            self.global_batch_size = self.batch_size * self.strategy.num_replicas_in_sync
            with self.strategy.scope():
                self.loss_fn: losses.Loss = HierarchicalContrastiveLoss(
                    alpha=self.alpha, temporal_unit=self.temporal_unit, reduction=losses.Reduction.NONE
                )
                self.optimizer: optimizers.Optimizer = eval(config.ts2vec.train.optimize_fn)(learning_rate=self.learning_rate)

        else:
            self.global_batch_size: int = None
            self.loss_fn: losses.Loss = HierarchicalContrastiveLoss(
                alpha=self.alpha, temporal_unit=self.temporal_unit
            )
            self.optimizer: optimizers.Optimizer = eval(config.ts2vec.train.optimize_fn)(learning_rate=self.learning_rate)

        self.current_step: None
        self.current_epoch: None
        self.result_state: Dict = {}

    def learn(self, inputs, verbose=1, project_path=None, save_path=None):
        """
        train
        :param inputs:
        :param verbose:
        :return:
        """

        assert tf.rank(inputs) == 3
        self.result_state['hierarchical_contrastive_loss_epoch'] = []
        self.result_state['hierarchical_contrastive_loss_iter'] = []
        train_dataset = self.create_train_dataset(inputs)

        if verbose == 1: print("=====" * 5, "Start learn")
        if self.iters is None and self.epochs is None:
            batches = train_dataset.cardinality().numpy()  # samples / batch_size
            self.iters = 200 if inputs.size <= 100000 else 600  # default param for n_iters
            self.epochs = (self.iters // batches) + 1 # TODO 맞는지 확인 + iter 기준 / epoch 기준 뭐로 할지 정하자..
        print('total iters: ', self.iters)
        print('total epochs: ', self.epochs)

        if project_path is None: project_path = hydra.utils.get_original_cwd()
        else: project_path = project_path
        mlflow.set_tracking_uri('file://'+str(project_path)+'/mlruns')
        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run():
            log_params_from_omegaconf_dict(self.config)
            self.current_epoch = 0
            self.current_iter = 0
            for epoch in tqdm(range(self.epochs), desc='learn', unit=' epoch'):
                if verbose == 1: print("\n", "=====" * 10, f"epoch {epoch + 1}: ")

                start_time = time.time()
                train_loss = self.train_one_epoch(train_dataset)

                if verbose == 1: print(f"train loss: {train_loss} \n"
                                       f"Time taken: %.2fs" % (time.time() - start_time))

            if save_path: self.model.save(Path(save_path, 'ts2vec'))

    def create_train_dataset(self, inputs: np.ndarray):
        """
        create tensorflow dataset as inputs of TS2Vec
        :param inputs: 3-dimensional numpy array (samples, timestamp, features)
        :return: tensorflow dataset
        """

        if self.max_train_length is not None:
            sections = inputs.shape[1] // self.max_train_length
            if sections >= 2:
                inputs = np.concatenate(split_with_nan(inputs, sections, axis=1), axis=0)

        temporal_missing = np.isnan(inputs).all(axis=-1).any(axis=0)
        if temporal_missing[0] or temporal_missing[-1]:  # 앞이나 뒤 time-series padding 이면,
            inputs = centerize_vary_length_series(inputs)  # center로 이동

        inputs = inputs[~np.isnan(inputs).all(axis=2).all(axis=1)] # 전체 temporal missing인 배치 삭제

        dataset = tf.data.Dataset.from_tensor_slices((inputs)).shuffle(
            buffer_size=self.buffer_size, seed=self.random_seed).batch(
            self.batch_size if self.strategy is None else self.global_batch_size).prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE)

        if self.strategy is None:
            return dataset
        else:
            dist_dataset = self.strategy.experimental_distribute_dataset(dataset)
            return dist_dataset

    def forward(self, subseries_inputs, crop_l, training=None):
        """
        forward function for TS2Vec.
        This method calculates model's forward inference value and empirical loss
        :param inputs: input_batch
        :return: empirical_loss
        """

        with tf.GradientTape() as self.tape:
            represent1 = self.model(subseries_inputs[0], training=training, mask='binomial')
            represent1 = represent1[:, -crop_l:] # TODO distribute에서 문제 없을지?/
            represent2 = self.model(subseries_inputs[1], training=training, mask='binomial')
            represent2 = represent2[:, :crop_l] # TODO tf.function 시 같은 timeseries 호출?
            if self.strategy is None:
                empirical_loss = self.loss_fn(represent1, represent2)
            else:
                per_example_loss = self.loss_fn(represent1, represent2)
                empirical_loss = tf.nn.compute_average_loss(per_example_loss, global_batch_size=self.global_batch_size)

        return empirical_loss

    def backward(self, empirical_loss):
        """
        backward backpropagation function for HierarchicalContrastiveLoss
        calculate the gradients with backpropagation
        :param empirical_loss: loss calcuated by HierarchcialContrastiveLoss
        :return: weight gradients
        """
        grads = self.tape.gradient(empirical_loss, self.model.trainable_variables)
        return grads

    def update(self, grads):
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    @tf.function # we cannot use graph mode since random cropping (after traincing, random crop lengths are constant)
    def train_one_step(self, inputs):
        """
        train TS2Vec one step with input batch
        :param inputs: input_batch
        :return: empirical loss
        """
        subseries_inputs, crop_l = RandomCropping(self.temporal_unit, inputs.shape)(inputs)
        empirical_loss = self.forward(subseries_inputs, crop_l, training=True) #TODO 이 안에서 random_crop 돌리면? (forward 밖, train_one_step 안)
        grads = self.backward(empirical_loss)
        self.update(grads)

        return empirical_loss

    @tf.function
    def distributed_train_step(self, inputs):
        per_replica_losses = self.strategy.run(self.train_one_step, args=(inputs, crop_l))
        return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

    def train_one_epoch(self, train_dataset: tf.data.Dataset):
        """
        train TS2Vec one epoch with Tensorflow dataset
        :param train_dataset: Tensorflow dataset
        :return:
        """
        cum_loss = 0
        epoch_iters = 0
        for step, input_batch in enumerate(train_dataset):
            if self.max_train_length is not None and input_batch.shape[1] > self.max_train_length:
                window_offset = tnp.random.randint(input_batch.shape[1] - self.max_train_length + 1, dtype=tnp.int32)
                input_batch = input_batch[:, window_offset:window_offset + self.max_train_length]  # TODO distribute에서 문제 없을지?

            if self.strategy is None: step_loss = self.train_one_step(input_batch)
            else: step_loss = self.distributed_train_step(input_batch)

            self.result_state["hierarchical_contrastive_loss_iter"].append(np.round(float(step_loss), 4))
            mlflow.log_metric(
                "hierarchical_contrastive_loss_iter",
                np.round(float(step_loss), 4), step=self.current_iter
            )

            cum_loss += step_loss
            epoch_iters += 1
            self.current_iter += 1
        self.current_epoch += 1
        train_loss = cum_loss / epoch_iters

        self.result_state["hierarchical_contrastive_loss_epoch"].append(np.round(float(train_loss), 4))
        mlflow.log_metric(
            "hierarchical_contrastive_loss_epoch",
            np.round(float(train_loss), 4), step=self.current_epoch
        )

        return train_loss

class EncodingProcedure(object):
    def __init__(self, model):
        self.model = model

    def encode(self, inputs, mask=None, encoding_window=None, casual=False, sliding_length=None, sliding_padding=0, batch_size=None):
        # assert self.net is not None, 'please train or load a net first'
        assert inputs.ndim == 3
        assert batch_size != None
        samples, ts_l, _ = inputs.shape

        dataset = tf.data.Dataset.from_tensor_slices((inputs)).batch(
            batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE) # TODO multi gpu 설정

        output = []
        for batch in dataset:
            if sliding_length is not None:
                represents = []
                if samples < batch_size:
                    calc_buffer, calc_buffer_l = [], 0

                for i in range(0, ts_l, sliding_length):
                    l = i - sliding_padding
                    r = i + sliding_length + (sliding_padding if not casual else 0)
                    batch_sliding = pad_nan(
                        batch[:, max(l, 0) : min(r, ts_l)],
                        left=-l if l < 0 else 0, right=r - ts_l if r > ts_l else 0,
                        axis=1
                    )
                    if samples < batch_size:
                        if calc_buffer_l + samples > batch_size:
                            out = self._eval_with_pooling(
                                tnp.concatenate(calc_buffer, axis=0)
                            )
                            represents += tnp.split(out, samples)
                            calc_buffer, calc_buffer_l = [], 0
                        calc_buffer.append(batch_sliding)
                        calc_buffer_l += samples
                    else:
                        out = self._eval_with_pooling(
                            batch_sliding, mask, slicing=slice(sliding_padding, sliding_padding + sliding_length),
                            encoding_window=encoding_window
                        )
                        represents.append(out)

                if samples < batch_size:
                    if calc_buffer_l > 0:
                        out = self._eval_with_pooling(
                            tnp.concatenate(calc_buffer, axis=0),# TODO eqal with torch.cat(calc_buffer, axis=0)?
                            mask,
                            slicing=slice(sliding_padding, sliding_padding + sliding_length),
                            encoding_window=encoding_window
                        )
                        represents += tnp.split(out, samples)
                        calc_buffer, calc_buffer_l = [], 0

                out = tnp.concatenate(represents, axis=1)

                if encoding_window == 'full_series':
                    out = layers.MaxPool1D(pool_size=out.shape[1])(out).sqeeze(1)
            else:
                out = self._eval_with_pooling(batch, mask, encoding_window=encoding_window)
                if encoding_window == 'full_series':
                    # out = out.squeeze(1)
                    out = tf.squeeze(out, axis=1)

            output.append(out)

        output = tnp.concatenate(output, axis=0)
        return output

    def _eval_with_pooling(self, inputs, mask=None, slicing=None, encoding_window=None):
        represent = self.model(inputs, mask=mask, training=False)

        if encoding_window == 'full_series':
            if slicing is not None:
                represent = represent[:, slicing]
            represent = layers.MaxPool1D(pool_size=represent.shape[1])(represent)

        elif isinstance(encoding_window, int):
            # represent = layers.MaxPool1D(pool_size=encoding_window, strirundes=1, padding = encoding_window // 2) # how to set padding size?
            represent = layers.MaxPool1D(pool_size=encoding_window, strides=1, padding='same')(
                represent)
            if encoding_window % 2 == 0:
                represent = represent[:, :-1]
            if slicing is not None:
                represent = represent[:, slicing]

        elif encoding_window == 'multiscale':
            p = 0
            represents = []
            while (1 << p) + 1 < represent.shape[1]:
                t_repressent = layers.MaxPool1D(
                    pool_size=(1 << (p + 1)) + 1, strides=1, padding='same'
                )(represent)  # TODO pdding=(1 << (p+1)) + 1 과 확인
                if slicing is not None:
                    t_repressent = t_repressent[:, slicing]
                represents.append(t_repressent)
                p += 1
            represent = layers.concatenate(represents, axis=-1)
        else:
            if slicing is not None:
                represent = represent[:, slicing]

        return represent

if __name__ == "__main__":

    import numpy as np
    from utils.learn_utils import centerize_vary_length_series
