import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from shutil import copy

import numpy as np
from keras import backend as K
from keras.models import load_model

from define import EN_GAME_COUNT, GAMEMAP_SHAPE, MAX_STEP, THREAD_NUMS
from puyo_game import State, make_puyo_list
from random_mcts import mcts_action

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def play(next_action_f, puyo_list):

    # 状態の生成
    state = State(np.zeros(GAMEMAP_SHAPE), puyo_list, 0)

    # game_map_image = plt.imshow(field_to_img(state.map))
    # plt.ion()
    reward = 0
    potential = 0
    while True:
        # game_map_image.set_data(field_to_img(state.map))
        # plt.pause(.01)
        if state.is_done():
            break

        potential = max(potential, state.calc_max_reward())
        # 行動の取得
        action = next_action_f(state)

        # 次状態の取得
        state, r = state.next(action)
        reward = max(reward, r)

    return (reward, potential)


def gready(state):
    next_action_value = 0
    action_list = []
    if len(state.puyo_list) <= 1:
        return random.choice(state.legal_actions())
    for next in state.legal_actions():
        next_state, _ = state.next(next)
        for dnext in next_state.legal_actions():
            dnext_state, _ = next_state.next(dnext)
            score = dnext_state.calc_max_reward()
            if score > next_action_value:
                action_list.clear()
                action_list.append(next)
                next_action_value = score
            elif score == next_action_value:
                action_list.append(next)
    if len(action_list) == 0:
        return random.choice(state.legal_actions())
    else:
        return random.choice(action_list)


def dual_play(t):
    puyo_list = make_puyo_list(MAX_STEP + 10)
    # 最新プレイヤー
    model0 = load_model(
        'C:/Users/rokahikou/Ohsuga_lab/AlphaPuyo/resources/latest.h5')
    next_action_f0 = mcts_action(model0)
    r0 = play(next_action_f0, puyo_list[:])
    K.clear_session()
    del model0
    # 出力
    print('\rEvaluate {}/{}\n'.format(t + 1, EN_GAME_COUNT), end='')
    return r0


def update_best_player():
    copy('C:/Users/rokahikou/Ohsuga_lab/AlphaPuyo/resources/latest.h5',
         'C:/Users/rokahikou/Ohsuga_lab/AlphaPuyo/resources/best.h5')
    print('Change BestPlayer')


def evaluate_network():
    workers = THREAD_NUMS

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                dual_play, t) for t in range(EN_GAME_COUNT)]
        result = [f.result() for f in as_completed(futures)]

    rewards = []
    pots = []

    for t in result:
        reward, pot = t

        rewards.append(reward)
        pots.append(pot)

    average = sum(rewards) / EN_GAME_COUNT
    average_p = sum(pots) / EN_GAME_COUNT

    with open('C:/Users/rokahikou/Ohsuga_lab/AlphaPuyo/evaluate.log', mode='a') as f:
        f.write('==AverageReward==\n')
        f.write('reward:{} / potential: {}\n'.format(average, average_p))
    update_best_player()


if __name__ == '__main__':
    evaluate_network()
