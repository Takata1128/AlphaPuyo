import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.callbacks import Callback, LambdaCallback, LearningRateScheduler
from keras.models import load_model
from keras.optimizers import SGD
from keras.utils import Sequence

from define import (BATCH_SIZE, CHANNEL, DN_OUTPUT_SIZE, HEIGHT, RN_EPOCHS,
                    WIDTH, DN_INPUT_SHAPE, GAMEMAP_HEIGHT, GAMEMAP_WIDTH, PUYO_COLOR, DATA_PATH, RESOURCE_PATH)


def encode(stage, puyos):
    return data2Binary(stage, puyos[0], puyos[1])


# データ変形用
def data2Binary(stage, nowpuyo, nextpuyo):
    index = stage
    binary = np.zeros(DN_INPUT_SHAPE)
    for i in range(0, GAMEMAP_HEIGHT):
        for j in range(0, GAMEMAP_WIDTH):
            num = index[i, j]
            binary[i, j, int(num)] = 1
    next_puyo_data = np.concatenate([nowpuyo, nextpuyo])
    for k in range(0, 4):
        for i in range(0, GAMEMAP_HEIGHT):
            for j in range(0, GAMEMAP_WIDTH):
                # binary[i, j, int(next_puyo_data[k]) + PUYO_COLOR] = 1
                binary[i, j, int(next_puyo_data[k]) +
                       PUYO_COLOR + k * PUYO_COLOR] = 1
    # for bit in range(0, TURN_CHANNELS):
    #     if turn & (1 << bit):
    #         for i in range(0, GAMEMAP_HEIGHT):
    #             for j in range(0, GAMEMAP_WIDTH):
    #                 binary[i, j, FIELD_CHANNELS + PUYO_CHANNELS + bit] = 1

    return binary


class TrainDataGenerator(Sequence):
    def __init__(self):
        self.data_paths = sorted(
            Path(DATA_PATH).glob('*.history'))
        self.length = len(self.data_paths)
        # self.max_v = 0
        # for item_path in self.data_paths:
        #     with item_path.open(mode='rb') as f:
        #         history = pickle.load(f)
        #         for h in history:
        #             self.max_v = max(self.max_v, h[3])

    def __getitem__(self, idx):
        item_path = self.data_paths[idx]
        with item_path.open(mode='rb') as f:
            history = pickle.load(f)

            train_data = np.zeros((BATCH_SIZE, HEIGHT, WIDTH, CHANNEL))
            p = np.zeros((BATCH_SIZE, DN_OUTPUT_SIZE))
            v = np.zeros(BATCH_SIZE)
            for i, h in enumerate(history):
                train_data[i] = encode(h[0], h[1])
                p[i] = h[2]
                v[i] = h[3]
            return train_data, [p, v]

    def __len__(self):
        return self.length

    def on_epoch_end(self):
        pass


class LossHistory(Callback):
    def __init__(self):
        # コンストラクタに保持用の配列を宣言しておく
        self.pi_loss = []
        self.v_loss = []

    def on_epoch_end(self, epoch, logs={}):
        # 配列にEpochが終わるたびにAppendしていく
        self.pi_loss.append(logs['pi_loss'])
        self.v_loss.append(logs['v_loss'])

        # グラフ描画部
        plt.figure(num=1, clear=True)
        plt.title('loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.plot(self.pi_loss, label='pi_loss')
        plt.plot(self.v_loss, label='v_loss')
        plt.legend()
        plt.pause(0.1)

    def on_train_end(self, logs={}):
        plt.savefig('loss.png')
        plt.close()


# 学習データの読み込み
def load_data():
    history_path = sorted(
        Path(DATA_PATH).glob('*.history'))[-1]
    with history_path.open(mode='rb') as f:
        return pickle.load(f)

# デュアルネットワークの学習


def train_network():
    train_gen = TrainDataGenerator()
    model = load_model(
        RESOURCE_PATH+'/best.h5')

    # モデルのコンパイル
    model.compile(loss=['categorical_crossentropy', 'mse'],
                  loss_weights={'pi': 0.8, 'v': 0.2},
                  optimizer=SGD(momentum=0.9))

    def step_decay(epoch):
        x = 0.002
        if epoch >= 10:
            x = 0.001
        if epoch >= 15:
            x = 0.0005
        return x

    lr_decay = LearningRateScheduler(step_decay)

    # 出力
    print_callback = LambdaCallback(on_epoch_begin=lambda epoch, logs: print(
        '\rTrain {}/{}'.format(epoch + 1, RN_EPOCHS), end=''))

    plot_callback = LossHistory()

    # 学習
    # model.fit(xs, [y_policies, y_values], batch_size=128,
    # epochs=RN_EPOCHS, verbose=0, callbacks=[lr_decay, print_callback,
    # plot_callback])
    model.fit_generator(train_gen, epochs=RN_EPOCHS, verbose=0, callbacks=[
                        lr_decay, print_callback, plot_callback])
    print('')

    # 最新プレイヤーのモデル保存
    model.save(RESOURCE_PATH+'/best.h5')

    K.clear_session()
    del model


if __name__ == '__main__':
    train_network()
