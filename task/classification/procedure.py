from typing import Dict
from omegaconf import DictConfig
from pathlib import Path
from tqdm import tqdm
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import losses, optimizers, metrics, models
import hydra
import mlflow

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from base.procedure import NetworkLearningProcedure, EncodingProcedure
from utils.utils import log_params_from_omegaconf_dict
from utils.learn_utils import CosineScheduler

class BaseNetworkLearn(NetworkLearningProcedure):
    def __init__(self, network, config, distribute_strategy):
        """
        This class is a module for learning network model.
        :param network: tensorflow Model
        :param config: omegaconf - DictConf
        """
        super().__init__(network, config, distribute_strategy)
        self.model: models.Model = network
        self.tape = None
        self.strategy = distribute_strategy

        self.config = config
        self.experiment_name = config.mlflow.experiment_name

        self.epochs: int = config.task.train.epochs
        self.random_seed: int = eval(config.random.random_seed)
        self.batch_size: int = config.task.train.batch_size
        self.buffer_size: int = config.task.train.buffer_size
        self.learning_rate: float = config.task.train.learning_rate
        self.early_stopping: bool = config.task.train.early_stopping
        self.stop_patience: int = config.task.train.stop_patience

        self.current_epoch = 0
        self.current_iter = 0

        if self.strategy is not None:
            self.global_batch_size = self.batch_size * self.strategy.num_replicas_in_sync
            with self.strategy.scope():
                self.loss_fn: losses.Loss = eval(config.task.train.loss_fn)(reduction=losses.Reduction.NONE)
                self.optimizer: optimizers.Optimizer = eval(config.task.train.optimize_fn)(learning_rate=self.learning_rate)
                self.loss_metric: tf.keras.metrics.Metric = eval(config.task.train.loss_metric)()
                self.result_metric: tf.keras.metrics.Metric = eval(config.task.train.evaluate_metric)()
        else:
            self.global_batch_size: int = None
            self.loss_fn: losses.Loss = eval(config.task.train.loss_fn)()
            self.optimizer: optimizers.Optimizer = eval(config.task.train.optimize_fn)(learning_rate=self.learning_rate)
            self.loss_metric: tf.keras.metrics.Metric = eval(config.task.train.loss_metric)()
            self.result_metric: tf.keras.metrics.Metric = eval(config.task.train.evaluate_metric)()
        self.result_state: Dict = None


    def learn(self, inputs, labels, valid_data=None, verbose=1, project_path=None, save_path=None):
        train_dataset = self.create_train_dataset(inputs, labels)
        if valid_data is not None: valid_dataset = self.create_valid_dataset(valid_data[0], valid_data[1])

        self.result_state = {}
        self.result_state[f'train_{self.loss_metric.name}'] = []
        self.result_state[f'train_{self.result_metric.name}'] = []
        self.result_state[f'valid_{self.loss_metric.name}'] = []
        self.result_state[f'valid_{self.result_metric.name}'] = []

        if project_path: pass
        else: project_path = hydra.utils.get_original_cwd()
        mlflow.set_tracking_uri('file://'+str(project_path)+'/mlruns')
        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run():
            log_params_from_omegaconf_dict(self.config)

            stop_wait = 0
            best = 0
            self.current_epoch = 0
            self.current_iter = 0
            for epoch in tqdm(range(self.epochs), desc='learn', unit=' epoch'):
                if verbose==1: print("\n", "=====" * 10, f"epoch {epoch + 1}: ")
                start_time = time.time()
                train_loss, train_eval = self.train_one_epoch(train_dataset)
                train_loss, train_eval = np.round(float(train_loss), 5), np.round(float(train_eval), 5)

                self.result_state[f'train_{self.loss_metric.name}'].append(train_loss)
                self.result_state[f'train_{self.result_metric.name}'].append(train_eval)

                if verbose==1: print(f"train {self.loss_metric.name}: {train_loss}, "
                                  f"train {self.result_metric.name}: {train_eval}")

                if valid_data is not None:
                    valid_loss, valid_eval = self.valid_one_epoch(valid_dataset)
                    valid_loss, valid_eval = np.round(float(valid_loss), 4), np.round(float(valid_eval), 4)

                    self.result_state[f'valid_{self.loss_metric.name}'].append(valid_loss)
                    self.result_state[f'valid_{self.result_metric.name}'].append(valid_eval)
                    if verbose==1: print(f"valid {self.loss_metric.name}: {valid_loss}, "
                                      f"valid {self.result_metric.name}: {valid_eval}")

                    # The early stopping strategy: stop the training if `val_loss` does not
                    # decrease over a certain number of epochs.
                    if self.early_stopping:
                        stop_wait += 1
                        if valid_loss > best:
                            best = valid_loss
                            stop_wait = 0
                        if stop_wait >= self.stop_patience:
                            print("=====" * 5, 'early stop', "=====" * 5)
                            break

                if verbose == 1: print("Time taken: %.2fs" % (time.time() - start_time))

            if save_path: self.model.save(Path(save_path, self.config.task.model.type))

    def create_train_dataset(self, inputs, labels):
        if self.strategy is None:
            dataset = tf.data.Dataset.from_tensor_slices((inputs, labels)).shuffle(
                buffer_size=self.buffer_size, seed=self.random_seed).batch(
                self.batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            return dataset
        else:  # self.strategy is not None
            dataset = tf.data.Dataset.from_tensor_slices((inputs, labels)).shuffle(
                buffer_size=self.buffer_size, seed=self.random_seed).batch(
                self.global_batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            dist_dataset = self.strategy.experimental_distribute_dataset(dataset)
            return dist_dataset

    def create_valid_dataset(self, inputs, labels):
        if self.strategy is None:
            dataset = tf.data.Dataset.from_tensor_slices((inputs, labels)).batch(self.batch_size).prefetch(
                buffer_size=tf.data.experimental.AUTOTUNE)
            return dataset
        else:  # self.strategy is not None
            dataset = tf.data.Dataset.from_tensor_slices((inputs, labels)).batch(self.global_batch_size).prefetch(
                buffer_size=tf.data.experimental.AUTOTUNE)
            dist_dataset = self.strategy.experimental_distribute_dataset(dataset)
            return dist_dataset

    def forward(self, inputs, labels):
        """
        Base forward inference function for tensorflow network.
        This method calculates model's forward inference value h and empirical loss
        :param input: server input data
        :return: inference value = intermediate vector h
        """
        with tf.GradientTape(persistent=True) as self.tape:
            predictions = self.model(inputs, training=True)
            if self.strategy is None:
                empirical_loss = tf.reduce_mean(self.loss_fn(labels, predictions))
            else:  #self.strategy is not None:
                per_example_loss = self.loss_fn(labels, predictions)
                empirical_loss = tf.nn.compute_average_loss(per_example_loss, global_batch_size=self.global_batch_size)
        return predictions, empirical_loss

    def backward(self, empirical_loss):
        """
        backward backpropagation function for Server network.
        calculate model's weight gradients with h gradient from client
        (dE/dh)*(dh/dw)=dE/dw
        :param h: intermediate vector h from server forward function
        :param h_grad_from_client: gradients of h from client backward function
        :return: weight gradients of clients model
        """
        grads = self.tape.gradient(empirical_loss, self.model.trainable_variables)
        return grads

    def update(self, grads):
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    @tf.function
    def train_one_step(self, inputs, labels):
        predictions, empirical_loss = self.forward(inputs, labels)
        grads = self.backward(empirical_loss)
        self.update(grads)
        self.loss_metric.update_state(y_true=labels, y_pred=predictions)
        self.result_metric.update_state(y_true=labels, y_pred=predictions)
        return empirical_loss

    @tf.function
    def valid_one_step(self, inputs, labels):
        predictions = self.model(inputs, training=False)
        self.loss_metric.update_state(y_true=labels, y_pred=predictions)
        self.result_metric.update_state(y_true=labels, y_pred=predictions)

    @tf.function
    def distributed_train_step(self, inputs, labels):
        per_replica_losses = self.strategy.run(self.train_one_step, args=(inputs, labels))
        return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

    @tf.function
    def distributed_valid_step(self, inputs, labels):
        self.strategy.run(self.valid_one_step, args=(inputs, labels,))

    def train_one_epoch(self, train_dataset:tf.data.Dataset):
        for step, (input_batch, label_batch) in enumerate(train_dataset):
            if self.strategy is None: self.train_one_step(input_batch, label_batch)
            else: self.distributed_train_step(input_batch, label_batch)
            # mlflow.log_metric(
            #     f'{self.loss_metric.name}_train_step',
            #     np.round(float(step_loss), 4), step=self.current_iter
            # ) TODO step 단위 log 추가

            self.current_iter += 1
        self.current_epoch += 1
        train_loss = self.loss_metric.result()
        train_eval = self.result_metric.result()
        self.loss_metric.reset_states()
        self.result_metric.reset_states()

        mlflow.log_metric(
            f'{self.loss_metric.name}_train_epoch',
            np.round(float(train_loss), 4), step=self.current_epoch
        )
        mlflow.log_metric(
            f'{self.result_metric.name}_train_epoch',
            np.round(float(train_eval), 4), step=self.current_epoch
        )

        return train_loss, train_eval

    def valid_one_epoch(self, valid_dataset):
        for input_batch, label_batch in valid_dataset:
            if self.strategy is None: self.valid_one_step(input_batch, label_batch)
            else: self.distributed_valid_step(input_batch, label_batch)
        valid_loss = self.loss_metric.result()
        valid_eval = self.result_metric.result()
        self.loss_metric.reset_states()
        self.result_metric.reset_states()

        mlflow.log_metric(
            f'{self.loss_metric.name}_valid_epoch',
            np.round(float(valid_loss), 4), step=self.current_epoch
        )
        mlflow.log_metric(
            f'{self.result_metric.name}_valid_epoch',
            np.round(float(valid_eval), 4), step=self.current_epoch
        )

        return valid_loss, valid_eval

    def create_callbacks(self):
        raise NotImplementedError()

class SVMLearn(object):
    def __init__(self, encoder, batch_size, search_type='grid'):
        self.encoder = encoder
        self.model = SVC(C=np.inf, gamma='scale')

        if search_type == 'grid':
            self.search_alg = GridSearchCV
        elif search_type == 'random':
            self.search_alg = RandomizedSearchCV

        self.batch_size = batch_size

    def create_search(self):
        self.search = self.search_alg(
            self.model, {
                'C': [
                    0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000,
                    np.inf
                ],
                'kernel': ['rbf'],
                'degree': [3],
                'gamma': ['scale'],
                'coef0': [0],
                'shrinking': [True],
                'probability': [False],
                'tol': [0.001],
                'cache_size': [200],
                'class_weight': [None],
                'verbose': [False],
                'max_iter': [10000000],
                'decision_function_shape': ['ovr'],
                'random_state': [None]
            },
            cv=5, n_jobs=5
        )


    def learn(self, inputs, labels, validation_data=None, verbose=1, project_path=None, save_path=None):
        encoding = EncodingProcedure(self.encoder)
        inputs = encoding.encode(inputs, encoding_window='full_series', batch_size=self.batch_size)

        if validation_data is not None:
            validation_data[0] = encoding.encode(validation_data[0], encoding_window='full_series')

        self.search.fit(inputs, labels, verbose=verbose)
        self.model = self.search.best_estimator_
        print('train accuracy: ', self.model.score(inputs, labels))
        print('valid accuracy: ', self.model.score(validation_data[0], validation_data[1]))


# def fit_svm(features, y, MAX_SAMPLES=10000):
#     nb_classes = np.unique(y, return_counts=True)[1].shape[0]
#     train_size = features.shape[0]
#
#     svm = SVC(C=np.inf, gamma='scale')
#     grid_search = GridSearchCV(
#         svm, {
#             'C': [
#                 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000,
#                 np.inf
#             ],
#             'kernel': ['rbf'],
#             'degree': [3],
#             'gamma': ['scale'],
#             'coef0': [0],
#             'shrinking': [True],
#             'probability': [False],
#             'tol': [0.001],
#             'cache_size': [200],
#             'class_weight': [None],
#             'verbose': [False],
#             'max_iter': [10000000],
#             'decision_function_shape': ['ovr'],
#             'random_state': [None]
#         },
#         cv=5, n_jobs=5
#     )
#     if train_size > MAX_SAMPLES:
#             split = train_test_split(
#                 features, y,
#                 train_size=MAX_SAMPLES, random_state=0, stratify=y
#             )
#             features = split[0]
#             y = split[2]
#
#     grid_search.fit(features, y)
#     return grid_search.best_estimator_
    # if train_size // nb_classes < 5 or train_size < 50:
    #     return svm.fit(features, y)
    # else:
    #     grid_search = GridSearchCV(
    #         svm, {
    #             'C': [
    #                 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000,
    #                 np.inf
    #             ],
    #             'kernel': ['rbf'],
    #             'degree': [3],
    #             'gamma': ['scale'],
    #             'coef0': [0],
    #             'shrinking': [True],
    #             'probability': [False],
    #             'tol': [0.001],
    #             'cache_size': [200],
    #             'class_weight': [None],
    #             'verbose': [False],
    #             'max_iter': [10000000],
    #             'decision_function_shape': ['ovr'],
    #             'random_state': [None]
    #         },
    #         cv=5, n_jobs=5
    #     )
    # If the training set is too large, subsample MAX_SAMPLES examples
    #     if train_size > MAX_SAMPLES:
    #         split = train_test_split(
    #             features, y,
    #             train_size=MAX_SAMPLES, random_state=0, stratify=y
    #         )
    #         features = split[0]
    #         y = split[2]
    #
    #     grid_search.fit(features, y)
    #     return grid_search.best_estimator_

# def eval_classification(model, train_data, train_labels, test_data, test_labels, eval_protocol='linear'):
#     assert train_labels.ndim == 1 or train_labels.ndim == 2
#     train_repr = model.encode(train_data, encoding_window='full_series' if train_labels.ndim == 1 else None)
#     test_repr = model.encode(test_data, encoding_window='full_series' if train_labels.ndim == 1 else None)
#
#     if eval_protocol == 'linear':
#         fit_clf = eval_protocols.fit_lr
#     elif eval_protocol == 'svm':
#         fit_clf = eval_protocols.fit_svm
#     elif eval_protocol == 'knn':
#         fit_clf = eval_protocols.fit_knn
#     else:
#         assert False, 'unknown evaluation protocol'
#
#     def merge_dim01(array):
#         return array.reshape(array.shape[0] * array.shape[1], *array.shape[2:])
#
#     if train_labels.ndim == 2:
#         train_repr = merge_dim01(train_repr)
#         train_labels = merge_dim01(train_labels)
#         test_repr = merge_dim01(test_repr)
#         test_labels = merge_dim01(test_labels)
#
#     clf = fit_clf(train_repr, train_labels)
#
#     acc = clf.score(test_repr, test_labels)
#     if eval_protocol == 'linear':
#         y_score = clf.predict_proba(test_repr)
#     else:
#         y_score = clf.decision_function(test_repr)
#     test_labels_onehot = label_binarize(test_labels, classes=np.arange(train_labels.max() + 1))
#     auprc = average_precision_score(test_labels_onehot, y_score)
#
#     return y_score, {'acc': acc, 'auprc': auprc}

