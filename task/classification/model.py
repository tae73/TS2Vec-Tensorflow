from tensorflow.keras import layers, models, regularizers
import numpy as np
from sklearn.svm import SVC
from base.network import TimeSeriesEncoder
from base.encoder import DilatedResidualBlock
from task.classification.procedure import BaseNetworkLearn, SVMLearn


class TSC(object):
    def __init__(self, cfg, encoder: TimeSeriesEncoder, distribute=None):
        self.cfg = cfg

        self.timesteps = cfg.data.timesteps
        self.features = cfg.data.features

        self.type = cfg.task.model.type
        self.tune_blocks = cfg.task.model.tune_blocks

        self.classes = cfg.data.classes

        self.encoder = encoder

        self.model = self.create_model()

        if self.type=='svm':
            self.learningprocedure = SVMLearn(self.encoder, cfg.task.train.batch_size, search_type='grid')
        else :
            self.learningprocedure = BaseNetworkLearn(self.model, cfg, distribute_strategy=distribute)


    def create_model(self):

        if self.type=='svm':
            self.model = SVC(C=np.inf, gamma='scale')

        elif self.type=='finetune' or self.type=='freeze' or self.type=='non_freeze':
            if self.type=='freeze':
                for layer in self.encoder.layers:
                    layer.trainable = False
            elif self.type == 'finetune':
                for layer in self.encoder.layers[:-(self.tune_blocks)]:
                    layer.trainable = False
            elif self.type == 'non_freeze':
                pass

            inputs = layers.Input((self.timesteps, self.features))
            represent = self.encoder(inputs, mask='all_true', training=False)
            maxpool = layers.MaxPool1D(pool_size=self.timesteps)(represent)
            dense = layers.Dense(units=self.cfg.ts2vec.model.hidden_size,
                                 kernel_regularizer=regularizers.l2(self.cfg.task.model.l2_decay),
                                 activation='gelu')(maxpool)
            if self.classes==1: outputs = layers.Dense(1, activation='sigmoid')(dense)
            else: outputs = layers.Dense(self.classes, activation='softmax')(dense)
            self.model = models.Model(inputs, outputs)

        elif self.type=='tcn_freeze' or self.type=='tcn_finetune' or self.type=='tcn_non_freeze':
            if self.type=='tcn_freeze':
                for layer in self.encoder.layers:
                    layer.trainable = False
            elif self.type=='tcn_finetune':
                for layer in self.encoder.layers[:-(self.tune_blocks)]:
                    layer.trainable = False
            elif self.type=='tcn_non_freeze':
                pass
            inputs = layers.Input((self.timesteps, self.features))
            represent = self.encoder(inputs, mask='all_true', training=False)
            tcn = DilatedResidualBlock(
                filters=self.cfg.ts2vec.model.hidden_size, kernel_size=3,
                dilation_rate=2**(self.cfg.ts2vec.model.depth+1), padding='causal',
                groups=1, activation='gelu', dropout=0, random_seed=None,
                batch_norm=False, layer_norm=False
                )(represent)
            maxpool = layers.MaxPool1D(pool_size=self.timesteps)(tcn)
            dense = layers.Dense(units=self.cfg.ts2vec.model.hidden_size//2,
                                 kernel_regularizer=regularizers.l2(self.cfg.task.model.l2_decay),
                                 activation='gelu')(maxpool)
            if self.classes==1: outputs = layers.Dense(1, activation='sigmoid')(dense)
            else: outputs = layers.Dense(self.classes, activation='softmax')(dense)
            self.model = models.Model(inputs, outputs)
        return self.model

    def learn(self, inputs, labels, valid_data, verbose=1, project_path=None, save_path=None):
         self.learningprocedure.learn(inputs, labels, valid_data, verbose, project_path, save_path)


