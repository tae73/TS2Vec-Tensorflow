import sys
sys.path.append('../TS2VEC')
from pathlib import Path
import hydra
from omegaconf import OmegaConf, DictConfig
import numpy as np

from base.network import TS2Vec
from task.classification.model import TSC

PROJECT_PATH = Path('.').absolute()
DATA_PATH = Path(PROJECT_PATH, 'data', 'eeg2')

def normalizer_3d(train_arr, target_arr):
    arr = train_arr.reshape(-1, train_arr.shape[2])
    std = arr.std(axis=0)
    mean = arr.mean(axis=0)
    return (target_arr - mean) / std

def load_eeg2(DATA_DIR):
    inputs_train = np.load(Path(DATA_DIR,'X_train.npy')).reshape(-1, 256, 64)
    inputs_test = np.load(Path(DATA_DIR, 'X_test.npy')).reshape(-1, 256, 64)
    labels_train = np.load(Path(DATA_DIR, 'y_train.npy'))
    labels_test = np.load(Path(DATA_DIR, 'y_test.npy'))

    inputs_train = normalizer_3d(inputs_train, inputs_train)
    inputs_test = normalizer_3d(inputs_train, inputs_test)
    return inputs_train, inputs_test, labels_train, labels_test

@hydra.main(config_path='.', config_name='cfg.yaml')
def main(cfg: DictConfig) -> None:
    # config = OmegaConf.load('./experiments/classification/eeg2/cfg.yaml')
    # os.system('mlflow ui --backend-store-uri' +
    #           'file://' + hydra.utils.get_original_cwd() + '/mlruns')
    # http://127.0.0.1:5000/

    inputs_train, inputs_test, labels_train, labels_test = load_eeg2(DATA_PATH)

    labels_train = labels_train.reshape(-1)
    labels_test = labels_test.reshape(-1)

    # learn TS2Vec
    ts2vec = TS2Vec(cfg)
    ts2vec.learn(inputs_train)

    # learn Classifier
    tsc = TSC(cfg, ts2vec.model)
    tsc.learn(inputs_train, labels_train.reshape(-1, 1, 1),
              valid_data=(inputs_test, labels_test.reshape(-1, 1, 1)))



if __name__ == "__main__":
    main()
