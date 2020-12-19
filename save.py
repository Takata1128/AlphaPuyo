import numpy as np
import os
import itertools
import pickle as cPickle
import shutil
import random
from datetime import datetime

GAMEMAP_HEIGHT = 13
GAMEMAP_WIDTH = 6
PUYOS_SHAPE = (2, 2)
GAMEMAP_SHAPE = (GAMEMAP_HEIGHT, GAMEMAP_WIDTH)


def chunks(list, n):
    for i in range(0, len(list), n):
        yield list[i:i + n]


def get_data(history):
    new_hist = []
    puyo_l = [1, 2, 3, 4]
    for h in history:
        orig_map = h[0]
        orig_puyo = h[1]
        for v in itertools.permutations(puyo_l):
            gamemap = np.zeros(GAMEMAP_SHAPE)
            puyos = np.zeros(PUYOS_SHAPE)
            for i in range(GAMEMAP_HEIGHT):
                for j in range(GAMEMAP_WIDTH):
                    if orig_map[i, j] != 0:
                        gamemap[i, j] = v[int(orig_map[i, j]) - 1]
            for i in range(2):
                for j in range(2):
                    puyos[i, j] = v[int(orig_puyo[i, j]) - 1]
            new_hist.append([gamemap, puyos, h[2], h[3]])
    return new_hist


def add_mirror(history):
    n = len(history)
    for k in range(n):
        orig_map = history[k][0]
        gamemap = np.zeros(GAMEMAP_SHAPE)
        for i in range(GAMEMAP_HEIGHT):
            for j in range(GAMEMAP_WIDTH):
                gamemap[i, j] = orig_map[i, GAMEMAP_WIDTH - j - 1]
        if gamemap[1, 2] == 0:  # Puyo.NONE
            history.append(
                [gamemap, history[k][1], history[k][2], history[k][3]])
    return history


def write_data(history):
    new_history = get_data(history)
    new_history = add_mirror(new_history)
    random.shuffle(new_history)
    chunk_history = chunks(new_history, 128)
    now = datetime.now()
    # if os.path.exists('./data/'):
    #     shutil.rmtree('./data/')
    os.makedirs('./data/', exist_ok=True)
    for i, c in enumerate(chunk_history):
        path = './data/{:04}{:02}{:02}{:02}_{}.history'.format(
            now.year, now.month, now.day, now.hour, i)
        with open(path, mode='wb') as f:
            cPickle.dump(c, f)
    # path = './data/{:04}{:02}{:02}{:02}{:02}{:02}.history'.format(
    #     now.year, now.month, now.day, now.hour, now.minute, now.second)
    # with open(path, mode='wb') as f:
    #     cPickle.dump(history, f)
