import sys
sys.path.append('../TS2VEC')
from pathlib import Path
import hydra
from omegaconf import OmegaConf, DictConfig
import numpy as np

from base.network import TS2Vec
from task.classification.model import TSC

PROJECT_PATH = Path('.').absolute()
DATA_PATH = Path(PROJECT_PATH, 'data', 'UCRArchive_2018', 'FordA')

def normalizer_3d(train_arr, target_arr):
    arr = train_arr.reshape(-1, train_arr.shape[2])
    std = arr.std(axis=0)
    mean = arr.mean(axis=0)
    return (target_arr - mean) / std

def load_forda(DATA_DIR):
    train = np.loadtxt(Path(DATA_DIR, 'FordA_TRAIN.tsv'))
    test = np.loadtxt(Path(DATA_DIR, 'FordA_TEST.tsv'))

    inputs_train = train[:, 1:]
    labels_train = train[:, 0]

    inputs_test = test[:, 1:]
    labels_test = test[:, 0]


    inputs_train = inputs_train.reshape((inputs_train.shape[0], inputs_train.shape[1], 1))
    inputs_test = inputs_test.reshape((inputs_test.shape[0], inputs_test.shape[1], 1))

    idx = np.random.permutation(len(inputs_train))
    inputs_train = inputs_train[idx]
    labels_train = labels_train[idx]

    labels_train[labels_train == -1] = 0
    labels_test[labels_test == -1] = 0

    # inputs_train = normalizer_3d(inputs_train, inputs_train)
    # inputs_test = normalizer_3d(inputs_train, inputs_test)
    return inputs_train, inputs_test, labels_train, labels_test

@hydra.main(config_path='.', config_name='cfg.yaml')
def main(cfg: DictConfig) -> None:
    # config = OmegaConf.load('./experiments/classification/fordA/cfg.yaml')
    # os.system('mlflow ui --backend-store-uri' +
    #           'file://' + hydra.utils.get_original_cwd() + '/mlruns')
    # http://127.0.0.1:5000/

    inputs_train, inputs_test, labels_train, labels_test = load_forda(DATA_PATH)

    # learn TS2Vec
    ts2vec = TS2Vec(cfg)
    ts2vec.learn(inputs_train)

    # learn Classifier
    tsc = TSC(cfg, ts2vec.model)
    tsc.learn(inputs_train, labels_train.reshape(-1, 1, 1),
              valid_data=(inputs_test, labels_test.reshape(-1, 1, 1)))



if __name__ == "__main__":
    # main()

    inputs_train, inputs_test, labels_train, labels_test = load_forda(DATA_PATH)
    cfg = OmegaConf.load('./experiments/classification/fordA/cfg.yaml')
    # learn TS2Vec
    ts2vec = TS2Vec(cfg)
    ts2vec.learn(inputs_train, project_path=PROJECT_PATH)

    cfg = OmegaConf.load('./experiments/classification/fordA/cfg_.yaml')


    tsc = TSC(cfg, ts2vec.model)
    tsc.learn(inputs_train, labels_train.reshape(-1, 1, 1), valid_data=(inputs_test, labels_test.reshape(-1, 1, 1)),
              verbose=1,
              project_path=PROJECT_PATH, save_path=None)

    # # learn Classifier
    # tsc = TSC(cfg, ts2vec.model)
    tsc.learn(inputs_train, labels_train.reshape(-1, 1, 1),
              valid_data=(inputs_test, labels_test.reshape(-1, 1, 1)), project_path=PROJECT_PATH)




