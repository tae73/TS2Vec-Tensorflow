from omegaconf import DictConfig
import tensorflow as tf
from tensorflow.keras import models
import numpy as np
from base.encoder import TimeSeriesEncoder
from base.procedure import NetworkLearningProcedure, HierarchicalContrastiveLearn, EncodingProcedure


class Network(object):
    def __init__(self, config: DictConfig):
        """
        Network class that creates and trains the tensorflow 2.x model and manages its parameters
        """
        self.model: models.Model = None
        self.learningprocedure: NetworkLearningProcedure = None
        self.strategy: tf.distribute.MirroredStrategy = None

    def create_model(self):
        raise NotImplementedError

    def learn(self):
        raise NotImplementedError

class TS2Vec(Network):

    def __init__(self, config: DictConfig, distribute=False):
        super(TS2Vec, self).__init__(config)

        self.depth = config.ts2vec.model.depth
        self.hidden_size = config.ts2vec.model.hidden_size
        self.out_size = config.ts2vec.model.out_size
        self.projection_activation = eval(config.ts2vec.model.projection_activation)
        self.kernel_size = config.ts2vec.model.kernel_size
        self.extractor_padding = config.ts2vec.model.extractor_padding
        self.extractor_activation = config.ts2vec.model.extractor_activation
        self.extractor_dropout_rate = config.ts2vec.model.extractor_dropout_rate
        self.extractor_batch_norm = config.ts2vec.model.extractor_batch_norm
        self.extractor_layer_norm = config.ts2vec.model.extractor_layer_norm
        self.represent_dropout_rate = config.ts2vec.model.represent_dropout_rate
        self.random_seed = eval(config.random.random_seed) if type(config.random.random_seed)==str else config.random.random_seed
        self.mask_node = config.ts2vec.model.mask_mode

        self.batch_size = config.ts2vec.train.batch_size
        self.timesteps = config.data.timesteps
        self.features = config.data.features

        self.model: models.Model = None
        self.strategy: tf.distribute.MirroredStrategy = None

        if not distribute:
            self.strategy = None
            self.model = self.create_network()
        else:
            self.strategy = tf.distribute.MirroredStrategy()
            with self.strategy.scope():
                self.model = self.create_network()
        self.learning_procedure = HierarchicalContrastiveLearn(self.model, config, self.strategy)
        self.encoding_procedure = EncodingProcedure(self.model)

    def create_network(self):

        self.model = TimeSeriesEncoder(
            depth=self.depth, hidden_size=self.hidden_size, out_size=self.out_size,
            projection_activation=self.projection_activation, kernel_size=self.kernel_size,
            extractor_padding=self.extractor_padding, extractor_activation=self.extractor_activation,
            extractor_dropout_rate=self.extractor_dropout_rate, extractor_batch_norm=self.extractor_batch_norm,
            extractor_layer_norm=self.extractor_layer_norm, represent_dropout_rate=self.represent_dropout_rate,
            mask_mode=self.mask_node, random_seed=self.random_seed
        )
        return self.model

    def learn(self, inputs, verbose=1, project_path=None, save_path=None):
        self.learning_procedure.learn(inputs, verbose, project_path, save_path)

    def encode(self, inputs, mask=None, encoding_window=None, casual=False, sliding_length=None, sliding_padding=0, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        output = self.encoding_procedure.encode(inputs, mask=mask, encoding_window=encoding_window, casual=casual,
                            sliding_length=sliding_length, sliding_padding=0, batch_size=batch_size)
        return output

    def compute_receptive_filed(self):
        raise NotImplementedError

    def show_model_structure(self):
        raise NotImplementedError

if __name__ == "__main__":
    from omegaconf import OmegaConf
    config = OmegaConf.load('./config/cfg.yaml')
    ts2vec = TS2Vec(config, TimeSeriesRepresentationLearn, EncodingProcedure)

    ts2vec.model.layers

    inputs = np.random.random((130, 30, 10)).astype(np.float32)
    import tensorflow as tf


    ts2vec.learn(inputs, verbose=1)

    from base.encoder import TemporalConv
    TemporalConv(channels=[64, 64, 64], kernel_size=3, padding='same', activation='gelu', dropout=0.0).layers



