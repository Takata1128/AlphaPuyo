import pickle as cPickle
import random

import numpy as np
from PIL import Image

from define import (ACTION_KIND, GAMEMAP_HEIGHT, GAMEMAP_WIDTH, MAX_STEP,
                    PUYO_COLOR, PUYOS_SHAPE)

actions_dict = {
    0: [1, 1, 0, 0, 0, 0],
    1: [2, 0, 0, 0, 0, 0],
    2: [2, 0, 0, 0, 0, 0],
    3: [1, 1, 0, 0, 0, 0],
    4: [0, 2, 0, 0, 0, 0],
    5: [0, 1, 1, 0, 0, 0],
    6: [0, 2, 0, 0, 0, 0],
    7: [0, 1, 1, 0, 0, 0],
    8: [0, 0, 2, 0, 0, 0],
    9: [0, 0, 1, 1, 0, 0],
    10: [0, 0, 2, 0, 0, 0],
    11: [0, 0, 1, 1, 0, 0],
    12: [0, 0, 0, 2, 0, 0],
    13: [0, 0, 0, 1, 1, 0],
    14: [0, 0, 0, 2, 0, 0],
    15: [0, 0, 0, 1, 1, 0],
    16: [0, 0, 0, 0, 2, 0],
    17: [0, 0, 0, 0, 1, 1],
    18: [0, 0, 0, 0, 2, 0],
    19: [0, 0, 0, 0, 1, 1],
    20: [0, 0, 0, 0, 0, 2],
    21: [0, 0, 0, 0, 0, 2],
}


class Puyo:
    NONE = 0
    RED = 1
    BLUE = 2
    GREEN = 3
    YELLOW = 4
    OJAMA = 5

    LEFT = 1
    RIGHT = 3
    UP = 2
    DOWN = 4

    def __init__(self, color1, color2, direction=UP):
        self.direct = direction
        self.color1 = color1
        self.color2 = color2

    def set_direct(self, direct):
        self.direct = direct


class State:
    def __init__(self, game_map, puyo_list, turn):
        self.map = game_map
        # if is_dead:
        #     self.puyos = np.zeros(PUYOS_SHAPE)
        # else:
        self.puyos = make_puyos(puyo_list)
        self.turn = turn
        self.puyo_list = puyo_list

    def is_done(self):
        return self.is_end() or self.is_lose()

    def is_lose(self):
        return self.map[1, 2] != Puyo.NONE

    def is_end(self):
        return self.turn == MAX_STEP

    def get_next_puyo(self):
        return self.puyo_list[0]

    # ぷよを１つだけ落とす
    def one_fall(self, stage, x, color):
        line = stage[:, x]
        if line[0] != Puyo.NONE:
            return stage, False
        noneIndex = np.where(line == Puyo.NONE)
        fall_y = np.max(noneIndex)
        stage[fall_y, x] = color
        is_alive = (stage[1, 2] == Puyo.NONE)
        return stage, is_alive

    # お邪魔を落とす（対戦用）
    def fall_ojama(self, ojama_num):
        if ojama_num == 0:
            return True
        base_num = ojama_num // 6
        ojama_num %= 6
        idx_list = [i for i in range(6)]
        extra_ojama = [0 for _ in range(6)]
        for e in random.sample(idx_list, ojama_num):
            extra_ojama[e] += 1

        for x in range(6):
            line = self.map[:, x]
            noneIndex = np.where(line == Puyo.NONE)[0]
            if len(noneIndex) == 0:
                continue
            fall_y = np.max(noneIndex)
            for j in range(base_num + extra_ojama[x]):
                if fall_y == -1:
                    break
                self.map[fall_y, x] = Puyo.OJAMA
                fall_y -= 1
        is_alive = (self.map[1, 2] == Puyo.NONE)
        return is_alive

    # 組ぷよを落とす
    def puyo_fall(self, stage, puyo, x):
        line = stage[:, x]
        try:
            if puyo.direct == Puyo.UP:
                # 親ぷよ
                NONEIndex = np.where(line == Puyo.NONE)
                fall_y = np.max(NONEIndex)
                stage[fall_y, x] = puyo.color1
                # 従属ぷよ
                line = stage[:, x]
                NONEIndex = np.where(line == Puyo.NONE)
                fall_y = np.max(NONEIndex)
                stage[fall_y, x] = puyo.color2

            elif puyo.direct == Puyo.LEFT:
                # 親ぷよ
                NONEIndex = np.where(line == Puyo.NONE)
                fall_y = np.max(NONEIndex)
                stage[fall_y, x] = puyo.color1
                # 従属ぷよ
                line = stage[:, x - 1]
                NONEIndex = np.where(line == Puyo.NONE)
                fall_y = np.max(NONEIndex)
                stage[fall_y, x - 1] = puyo.color2

            elif puyo.direct == Puyo.RIGHT:
                # 親ぷよ
                NONEIndex = np.where(line == Puyo.NONE)
                fall_y = np.max(NONEIndex)
                stage[fall_y, x] = puyo.color1
                # 従属ぷよ
                line = stage[:, x + 1]
                NONEIndex = np.where(line == Puyo.NONE)
                fall_y = np.max(NONEIndex)
                stage[fall_y, x + 1] = puyo.color2

            elif puyo.direct == Puyo.DOWN:
                # 従属ぷよ
                NONEIndex = np.where(line == Puyo.NONE)
                fall_y = np.max(NONEIndex)
                stage[fall_y, x] = puyo.color2
                # 親ぷよ
                NONEIndex = np.where(line == Puyo.NONE)
                fall_y = np.max(NONEIndex)
                stage[fall_y, x] = puyo.color1
        except(ValueError):
            return stage, False

        # 死亡判定
        is_alive = (stage[1, 2] == Puyo.NONE)
        return stage, is_alive

    # ぷよ消しのアルゴリズム
    def erase(self, stage, newStage):
        diff = stage - newStage
        getScore = 0.0
        points = np.where(diff != 0)
        for i in range(0, len(points[0])):
            try:
                x = points[1][i]
                y = points[0][i]
            except IndexError:
                print('!?')
            color = newStage[y, x]
            cpStage, counter = self.erase_puyo(
                cPickle.loads(cPickle.dumps(newStage, -1)), x, y, color)
            if counter >= 4:
                getScore += counter * 5
                newStage = cpStage
        return newStage, getScore

    def erase_puyo(self, stage, x, y, color, counter=0):
        if counter == 0 and color == Puyo.OJAMA:
            return stage, counter
        if color == Puyo.NONE:
            return stage, counter
        if stage[y, x] == Puyo.OJAMA:
            stage[y, x] = Puyo.NONE
            return stage, counter
        if stage[y, x] != color:
            return stage, counter
        else:
            counter += 1
            stage[y, x] = Puyo.NONE

        if y - 1 != -1:
            stage, counter = self.erase_puyo(stage, x, y - 1, color, counter)
        if y + 1 != GAMEMAP_HEIGHT:
            stage, counter = self.erase_puyo(stage, x, y + 1, color, counter)
        if x - 1 != -1:
            stage, counter = self.erase_puyo(stage, x - 1, y, color, counter)
        if x + 1 != GAMEMAP_WIDTH:
            stage, counter = self.erase_puyo(stage, x + 1, y, color, counter)
        return stage, counter

    def erase_simulation(self, game_map, new_game_map):
        counter = 0
        reward = 0
        while(True):
            # 消す処理
            new_stage, get_score = self.erase(cPickle.loads(cPickle.dumps(
                game_map, -1)), cPickle.loads(cPickle.dumps(new_game_map, -1)))
            if get_score > 0:
                counter += 1
                reward += 1
            elif get_score == 0:
                game_map = new_stage
                break

            # 落とす処理
            # game_map: 落とす前,　new_game_map:　落とした後
            new_game_map = self.fall(
                cPickle.loads(cPickle.dumps(new_stage, -1)))
            game_map = new_stage
        return game_map, reward

    # ぷよを落下させる

    def fall(self, stage):
        for i in range(GAMEMAP_WIDTH):
            while(True):
                line = stage[:, i]
                NONEIndex = np.where(line == Puyo.NONE)
                PUYOIndex = np.where(line != Puyo.NONE)

                if len(NONEIndex[0]) == 0 or len(PUYOIndex[0]) == 0:
                    break

                try:
                    noneMax = np.max(NONEIndex)
                    puyoMin = np.min(PUYOIndex)
                except ValueError:
                    print('!?')

                if (noneMax > puyoMin):
                    line[puyoMin + 1:noneMax + 1] = line[puyoMin:noneMax]
                    line[puyoMin] = Puyo.NONE
                    # print(puyoMin,noneMax)
                    stage[:, i] = line
                else:
                    break

        return stage

    def legal_actions(self):
        actions = []
        global actions_dict
        for action in range(0, ACTION_KIND):
            ok = True
            for x in range(GAMEMAP_WIDTH):
                if np.count_nonzero(
                        self.map[:, x] == Puyo.NONE) < actions_dict[action][x]:
                    ok = False
            if ok:
                actions.append(action)
        return actions

    def next(self, action):
        game_map = cPickle.loads(cPickle.dumps(self.map, -1))
        puyo_list = self.puyo_list[:]
        # if len(puyo_list) == 0:
        #     return State(np.zeros(GAMEMAP_SHAPE), puyo_list), 0
        puyo = puyo_list[0]
        puyo_list.pop(0)  # 次状態の準備
        reward = 0
        # 向き・落下点決め
        if action < 3:
            x = 0
            if action == 0:
                puyo.direct = puyo.RIGHT
            if action == 1:
                puyo.direct = puyo.DOWN
            if action == 2:
                puyo.direct = puyo.UP
        elif action > 18:
            if action == 19:
                puyo.direct = puyo.LEFT
            if action == 20:
                puyo.direct = puyo.DOWN
            if action == 21:
                puyo.direct = puyo.UP
            x = 5
        else:
            puyo.direct = ((action + 1) - 4) % 4 + 1
            x = int((action - 3) / 4) + 1
        new_game_map, flag = self.puyo_fall(
            cPickle.loads(cPickle.dumps(game_map, -1)), puyo, x)

        # 死んだとき
        # if not flag:
        #     return State(res_map, puyo_list),
        # else:
        res_map, reward = self.erase_simulation(game_map, new_game_map)
        return State(res_map, puyo_list, self.turn + 1), reward

    def calc_next_reward(self, action):
        game_map = cPickle.loads(cPickle.dumps(self.map, -1))
        if len(self.puyo_list) == 0:
            return 0
        puyo = self.puyo_list[0]
        reward = 0
        # 向き・落下点決め
        if action < 3:
            x = 0
            if action == 0:
                puyo.direct = puyo.RIGHT
            if action == 1:
                puyo.direct = puyo.DOWN
            if action == 2:
                puyo.direct = puyo.UP
        elif action > 18:
            if action == 19:
                puyo.direct = puyo.LEFT
            if action == 20:
                puyo.direct = puyo.DOWN
            if action == 21:
                puyo.direct = puyo.UP
            x = 5
        else:
            puyo.direct = ((action + 1) - 4) % 4 + 1
            x = int((action - 3) / 4) + 1
        new_game_map, flag = self.puyo_fall(
            cPickle.loads(cPickle.dumps(game_map, -1)), puyo, x)

        # 死んだとき
        if not flag:
            return 0
        else:
            _, reward = self.erase_simulation(game_map, new_game_map)
            return reward

    def calc_max_reward(self):
        old_map = cPickle.loads(cPickle.dumps(self.map, -1))
        reward = 0
        for color in range(1, PUYO_COLOR + 1):
            for x in range(0, GAMEMAP_WIDTH):
                game_map = cPickle.loads(cPickle.dumps(self.map, -1))
                new_gamemap, is_alive = self.one_fall(game_map, x, color)
                if not is_alive:
                    continue
                _, tmp = self.erase_simulation(old_map, new_gamemap)
                reward = max(reward, tmp)
        return reward


def make_puyos(puyo_list):
    puyos = np.zeros(PUYOS_SHAPE)
    if len(puyo_list) == 0:
        puyos[0][0] = 0
        puyos[0][1] = 0
        puyos[1][0] = 0
        puyos[1][1] = 0

    elif len(puyo_list) == 1:
        puyos[0][0] = puyo_list[0].color1
        puyos[0][1] = puyo_list[0].color2
        puyos[1][0] = 0
        puyos[1][1] = 0

    else:
        puyos[0][0] = puyo_list[0].color1
        puyos[0][1] = puyo_list[0].color2
        puyos[1][0] = puyo_list[1].color1
        puyos[1][1] = puyo_list[1].color2
    return puyos


def make_puyo_list(step):
    puyo_list = []
    for i in range(step):
        puyo_list.append(Puyo(random.randint(1, PUYO_COLOR),
                              random.randint(1, PUYO_COLOR)))
    return puyo_list


def field_to_img(field):
    green = Image.open("img/green.png")
    yellow = Image.open("img/yellow.png")
    blue = Image.open("img/blue.png")
    red = Image.open("img/red.png")
    ojama = Image.open("img/ojama.png")
    blank = Image.open("img/blank.png")
    imgs = [green, yellow, blue, red, ojama, blank]
    color_type = [1, 2, 3, 4, 5, 0]
    field_img = Image.new(
        "RGB",
        (green.width *
         GAMEMAP_WIDTH,
         green.height *
         GAMEMAP_HEIGHT))
    start_y = 0
    for y in field:
        field_x_img = Image.new(
            "RGB", (green.width * GAMEMAP_WIDTH, green.height))
        start_x = 0
        for x in y:
            for img, color in zip(imgs, color_type):
                if x == color:
                    field_x_img.paste(img, (start_x, 0))
                    start_x += img.width
        field_img.paste(field_x_img, (0, start_y))
        start_y += field_x_img.height
    return field_img
