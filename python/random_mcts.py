import random
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model

from define import (ACTION_KIND, CHANNEL_SIZE, DN_INPUT_SHAPE, EVALUATE_DEPTH,
                    FIELD_CHANNELS, GAMEMAP_HEIGHT, GAMEMAP_SHAPE,
                    GAMEMAP_WIDTH, MAX_STEP, PUYO_CHANNELS, PUYO_COLOR,
                    PV_EVALUATE_COUNT, TRY_COUNT, TSUMO_SIZE, TURN_CHANNELS, RESOURCE_PATH)
from puyo_game import Puyo, State, actions_dict, field_to_img

# patterns_dict = {
#     [1, 1]: 0,
#     [1, 2]: 1,
#     [1, 3]: 2,
#     [1, 4]: 3,
#     [2, 1]: 1,
#     [2, 2]: 4,
#     [2, 3]: 5,
#     [2, 4]: 6,
#     [3, 1]: 2,
#     [3, 2]: 5,
#     [3, 3]: 7,
#     [3, 4]: 8,
#     [4, 1]: 3,
#     [4, 2]: 6,
#     [4, 3]: 8,
#     [4, 4]: 9
# }

# ノードのリストを試行回数のリストに変換


def nodes_to_scores(nodes):
    scores = []
    for c in nodes:
        scores.append(c.n)
    return scores


def encode(stage, puyos, turn):
    return data2Binary(stage, puyos[0], puyos[1], turn)


# データ変形用
def data2Binary(stage, nowpuyo, nextpuyo, turn):
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


# バッチ形式に変形する
def state2batch(gameMap, batch_size=1):
    return gameMap.reshape(
        (batch_size,
         GAMEMAP_HEIGHT,
         GAMEMAP_WIDTH,
         CHANNEL_SIZE))


def predict(model, state):
    y = model.predict(state2batch(encode(state.map, state.puyos, state.turn)))
    policies = y[0][0][list(state.legal_actions())]  # 合法手のみ
    policies /= sum(policies) if sum(policies) else 1  # 合計１の確率分布に変換
    values = y[1][0][0]
    return policies, values


def mcts_scores(model, state, temperature):
    def make_puyo_list(next2):
        ret = []
        ret.append(Puyo(next2[0][0], next2[0][1]))
        ret.append(Puyo(next2[1][0], next2[1][1]))
        while len(ret) < TSUMO_SIZE:
            next = Puyo(random.randint(1, 4), random.randint(1, 4))
            ret.append(next)
        return ret

    def legal_actions(field):
        actions = []
        for action in range(0, ACTION_KIND):
            ok = True
            for x in range(GAMEMAP_WIDTH):
                if np.count_nonzero(
                        field[:, x] == Puyo.NONE) < actions_dict[action][x]:
                    ok = False
            if ok:
                actions.append(action)
        return actions

    # モンテカルロ木探索のノードの定義
    class node:
        def __init__(self, state, d, p, r):
            self.state = state
            self.d = d
            self.p = p
            self.r = r
            self.w = 0
            self.n = 0
            self.child_nodes = None

        def evaluate(self):
            # ゲーム終了時
            if self.state.is_done():
                value = -1 if self.state.is_lose() else self.r
                # 累計価値と試行回数の更新
                self.w += value
                self.n += 1
                return value

            # 一定の深さに達したら
            if self.d == EVALUATE_DEPTH:
                value = self.state.calc_max_reward()
                self.w += value
                self.n += 1
                return value

            # まだ展開していない
            if not self.child_nodes:
                policies, value = predict(model, self.state)
                # 累計価値と試行回数の更新
                self.w += value
                self.n += 1
                # 子ノードの展開
                self.child_nodes = []
                for action, policy in zip(
                        self.state.legal_actions(), policies):
                    next_state, r = self.state.next(action)
                    self.child_nodes.append(
                        node(next_state, self.d + 1, policy, max(self.r, r)))
                return value

            # 子ノードが存在するとき
            else:
                # アーク評価値が最大の子ノードの評価で価値を取得
                value = self.next_child_node().evaluate()
                # 累計価値と試行回数更新
                self.w += value
                self.n += 1
                return value

        # アーク評価値最大の子ノードを取得

        def next_child_node(self):
            # アーク評価値の計算
            t = sum(nodes_to_scores(self.child_nodes))
            pucb_values = []
            for child_node in self.child_nodes:
                pucb_values.append((child_node.w /
                                    child_node.n if child_node.n else 0.0) +
                                   (self.w / self.n) *
                                   child_node.p *
                                   sqrt(t) /
                                   (1 +
                                    child_node.n))

            return self.child_nodes[np.random.choice(
                np.flatnonzero(pucb_values == max(pucb_values)))]

    # 複数ツモで評価をトライ
    scores = [0 for i in range(len(legal_actions(state.map)))]
    for _ in range(TRY_COUNT):
        # 適当にツモを生成し複数回評価
        puyo_list = make_puyo_list(state.puyos)
        S = State(state.map, puyo_list, state.turn)
        root_node = node(S, 0, 0, 0)
        for __ in range(PV_EVALUATE_COUNT):
            root_node.evaluate()
        for i in range(len(root_node.child_nodes)):
            scores[i] += root_node.child_nodes[i].n

    # 合法手の確率分布
    if temperature == 0:  # 最大値のみ1
        action = np.argmax(scores)
        scores = np.zeros(len(scores))
        scores[action] = 1
    else:  # ボルツマン分布でバラつき付加
        scores = boltzman(scores, temperature)

    return scores


def normalize_value(value):
    return 1 - 1 / value if value != 0 else 0


# モンテカルロ木探索で行動選択
def mcts_action(model, temperature=0):
    def mcts_action(state):
        scores = mcts_scores(model, state, temperature)
        return np.random.choice(state.legal_actions(), p=scores)
    return mcts_action


def predict_action(model):
    def predict_action(state):
        y = model.predict(state2batch(encode(state.map, state.puyos)))
        policies = y[0][0][list(state.legal_actions())]  # 合法手のみ
        policies /= sum(policies) if sum(policies) else 1  # 合計１の確率分布に変換
        return np.argmax(policies)
    return predict_action


# ボルツマン分布
def boltzman(xs, temperature):
    xs = [x**(1 / temperature) for x in xs]
    return [x / sum(xs) for x in xs]


if __name__ == '__main__':
    model = load_model(RESOURCE_PATH+'/best.h5')
    puyo_list = list()
    for i in range(MAX_STEP):
        puyo_list.append(
            Puyo(random.randint(1, PUYO_COLOR), random.randint(1, PUYO_COLOR)))
    state = State(np.zeros(GAMEMAP_SHAPE), puyo_list, 0)
    next_action = mcts_action(model, 1.0)

    game_map_image = plt.imshow(field_to_img(state.map))
    plt.ion()
    while True:
        game_map_image.set_data(field_to_img(state.map))
        plt.pause(.01)
        rensa = state.calc_max_reward()
        print('\r{} : {}'.format(state.turn, rensa))
        if state.is_done():
            break

        # 行動の取得
        action = next_action(state)

        # 次状態の取得
        state, r = state.next(action)
    print('please any input')
    input()
