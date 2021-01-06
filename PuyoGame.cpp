#include "puyoGame.hpp"
#include <cassert>
#include <cstddef>
#include <queue>
#include <random>

const int dx[] = {0, -1, 0, 1};
const int dy[] = {1, 0, -1, 0};

puyogame::Puyo::Puyo(int color1, int color2) {
    this->direction = UP;
    this->color1 = color1;
    this->color2 = color2;
}

void puyogame::Puyo::set_direct(int direct) { this->direction = direct; }

puyogame::State::State() {
    this->gameMap = VVI();
    this->puyos = VVI(2, VI(2));
    this->turn = 0;
}

puyogame::State::State(VVI gameMap, VVI puyos, int turn) {
    this->gameMap = gameMap;
    this->puyos = puyos;
    this->turn = turn;
}

bool puyogame::State::isDone() { return this->isEnd() || this->isLose(); }

bool puyogame::State::isLose() {
    return this->gameMap[DEAD_Y][DEAD_X] != puyogame::Puyo::NONE;
}

bool puyogame::State::isEnd() { return this->turn == MAX_STEP; }

puyogame::Puyo puyogame::State::getNextPuyo() {
    return Puyo(puyos[0][0], puyos[0][1]);
}

inline int getFallY(const VVI &stage, int x) {
    size_t fallY = 0;
    while(fallY + 1 < GAMEMAP_HEIGHT &&
          stage[fallY + 1][x] == puyogame::Puyo::NONE)
        fallY++;
    return fallY;
}

VVI puyogame::State::oneFall(VVI stage, size_t x, int color, bool &isAlive) {
    assert(stage[0][x] == puyogame::Puyo::NONE);

    size_t fallY = getFallY(stage, x);
    stage[fallY][x] = color;
    isAlive = (stage[1][2] == puyogame::Puyo::NONE);
    return stage;
}

// 組ぷよを落とす
VVI puyogame::State::puyoFall(VVI stage, size_t x, puyogame::Puyo puyo,
                              bool &isAlive) {
    if(puyo.direction == puyogame::Puyo::UP) {
        assert(stage[0][x] == puyogame::Puyo::NONE &&
               stage[1][x] == puyogame::Puyo::NONE);
        size_t fallY = getFallY(stage, x);
        stage[fallY][x] = puyo.color1;
        fallY = getFallY(stage, x);
        stage[fallY][x] = puyo.color2;
    } else if(puyo.direction == puyogame::Puyo::RIGHT) {
        assert(x + 1 < GAMEMAP_WIDTH);
        assert(stage[0][x] == puyogame::Puyo::NONE &&
               stage[0][x + 1] == puyogame::Puyo::NONE);
        size_t fallY = getFallY(stage, x);
        stage[fallY][x] = puyo.color1;
        fallY = getFallY(stage, x + 1);
        stage[fallY][x + 1] = puyo.color2;
    } else if(puyo.direction == puyogame::Puyo::DOWN) {
        assert(stage[0][x] == puyogame::Puyo::NONE &&
               stage[1][x] == puyogame::Puyo::NONE);
        size_t fallY = getFallY(stage, x);
        stage[fallY][x] = puyo.color2;
        fallY = getFallY(stage, x);
        stage[fallY][x] = puyo.color1;
    } else if(puyo.direction == puyogame::Puyo::LEFT) {
        assert(x - 1 >= 0);
        assert(stage[0][x] == puyogame::Puyo::NONE &&
               stage[0][x - 1] == puyogame::Puyo::NONE);
        size_t fallY = getFallY(stage, x);
        stage[fallY][x] = puyo.color1;
        fallY = getFallY(stage, x - 1);
        stage[fallY][x - 1] = puyo.color2;
    }
    isAlive = (stage[DEAD_Y][DEAD_X] == puyogame::Puyo::NONE);
    return stage;
}

VVI puyogame::State::erase(VVI stage, VVI newStage, int &getScore) {
    getScore = 0;
    std::vector<std::pair<int, int>> vec;
    for(int i = 0; i < GAMEMAP_HEIGHT; i++) {
        for(int j = 0; j < GAMEMAP_WIDTH; j++) {
            if(stage[i][j] - newStage[i][j] != 0) {
                vec.emplace_back(j, i);
            }
        }
    }
    for(auto p : vec) {
        int x = p.first, y = p.second;
        int color = newStage[y][x];
        int counter = 0;
        VVI cpStage = this->erasePuyo(newStage, x, y, color, counter);
        if(counter >= 4) {
            getScore += counter * 5;
            newStage = cpStage;
        }
    }
    return newStage;
}

VVI puyogame::State::erasePuyo(VVI stage, size_t sx, size_t sy, int color,
                               int &counter) {
    counter = 0;
    if(color == puyogame::Puyo::OJAMA) {
        return stage;
    }
    if(color == puyogame::Puyo::NONE) {
        return stage;
    }
    std::queue<std::pair<int, int>> que;
    que.emplace(sx, sy);
    while(!que.empty()) {
        auto [x, y] = que.front();
        que.pop();
        if(stage[y][x] == puyogame::Puyo::OJAMA) {
            stage[y][x] = puyogame::Puyo::NONE;
            continue;
        }
        counter++;
        stage[y][x] = puyogame::Puyo::NONE;
        for(int i = 0; i < 4; i++) {
            int nx = x + dx[i], ny = y + dy[i];
            if(nx >= 0 && nx < GAMEMAP_WIDTH && ny >= 0 &&
               ny < GAMEMAP_HEIGHT) {
                if(stage[ny][nx] == puyogame::Puyo::OJAMA ||
                   stage[ny][nx] == color) {
                    que.emplace(nx, ny);
                }
            }
        }
    }
    return stage;
}

VVI puyogame::State::eraseSimulation(VVI gameMap, VVI newGameMap, int &reward) {
    int counter = 0;
    reward = 0;
    while(1) {
        // 消す
        int getScore = 0;
        VVI newStage = this->erase(gameMap, newGameMap, getScore);
        if(getScore > 0) {
            counter++;
            reward++;
        } else if(getScore == 0) {
            gameMap = newStage;
            break;
        }

        // 落とす
        newGameMap = this->fall(newStage);
        gameMap = newStage;
    }
    return gameMap;
}

VVI puyogame::State::fall(VVI stage) {
    for(int x = 0; x < GAMEMAP_WIDTH; x++) {
        std::queue<int> que;
        for(int y = GAMEMAP_HEIGHT - 1; y >= 0; y--) {
            if(stage[y][x] != puyogame::Puyo::NONE) {
                que.push(stage[y][x]);
            }
        }
        for(int y = GAMEMAP_HEIGHT - 1; y >= 0; y--) {
            if(que.empty()) {
                stage[y][x] = puyogame::Puyo::NONE;
            } else {
                stage[y][x] = que.front();
                que.pop();
            }
        }
    }
    return stage;
}

VI puyogame::State::legalActions() {
    VI ret;
    int actionIdx = 0;
    for(auto action_vec : ACTIONSDICT) {
        bool ok = true;
        for(int x = 0; x < GAMEMAP_WIDTH; x++) {
            int cnt = 0;
            for(int y = 0; y < GAMEMAP_HEIGHT; y++) {
                if(this->gameMap[y][x] == puyogame::Puyo::NONE)
                    cnt++;
            }
            if(cnt < action_vec[x])
                ok = false;
        }
        if(ok)
            ret.push_back(actionIdx);
        actionIdx++;
    }
    return ret;
}

inline VVI makePuyos(const VI &puyo1, const VI &puyo2) {
    VVI ret = VVI(2, VI(2));
    ret[0][0] = puyo1[0];
    ret[0][1] = puyo1[1];
    ret[1][0] = puyo2[0];
    ret[1][1] = puyo2[1];
    return ret;
}

// // 次状態に遷移
// puyogame::State puyogame::State::next(int action, VI nextPuyoColors,
//                                       int &reward) {
//     VVI gameMap = this->gameMap;
//     Puyo puyo = getNextPuyo();
//     int x;
//     if(action < 3) {
//         x = 0;
//         if(action == 0)
//             puyo.direction = puyo.RIGHT;
//         if(action == 1)
//             puyo.direction = puyo.DOWN;
//         if(action == 2)
//             puyo.direction = puyo.UP;
//     } else if(action > 18) {
//         x = 5;
//         if(action == 19)
//             puyo.direction = puyo.LEFT;
//         if(action == 20)
//             puyo.direction = puyo.DOWN;
//         if(action == 21)
//             puyo.direction = puyo.UP;
//     } else {
//         puyo.direction = ((action + 1) - 4) % 4 + 1;
//         x = ((action - 3) / 4) + 1;
//     }

//     bool isAlive = 1;
//     VVI newGameMap = this->puyoFall(gameMap, x, puyo, isAlive);
//     VVI resMap = this->eraseSimulation(gameMap, newGameMap, reward);
//     VVI nextPuyos = makePuyos(puyos[1], nextPuyoColors);
//     return State(resMap, nextPuyos, this->turn + 1);
// }

// 次状態に遷移
puyogame::State puyogame::State::next(int action, VI nextPuyoColors,
                                      int &reward) {
    VVI gameMap = this->gameMap;
    Puyo puyo = getNextPuyo();
    int x;
    if(action < 3) {
        x = 0;
        if(action == 0)
            puyo.direction = puyo.RIGHT;
        if(action == 1)
            puyo.direction = puyo.DOWN;
        if(action == 2)
            puyo.direction = puyo.UP;
    } else if(action > 10) {
        x = 3;
        if(action == 11)
            puyo.direction = puyo.LEFT;
        if(action == 12)
            puyo.direction = puyo.DOWN;
        if(action == 13)
            puyo.direction = puyo.UP;
    } else {
        puyo.direction = ((action + 1) - 4) % 4 + 1;
        x = ((action - 3) / 4) + 1;
    }
    bool isAlive = 1;
    VVI newGameMap = this->puyoFall(gameMap, x, puyo, isAlive);
    VVI resMap = this->eraseSimulation(gameMap, newGameMap, reward);
    VVI nextPuyos = makePuyos(puyos[1], nextPuyoColors);
    return State(resMap, nextPuyos, this->turn + 1);
}

int puyogame::State::calcMaxReward() {
    auto gameMap = this->gameMap;
    int reward = 0;
    for(int color = 1; color <= PUYO_COLOR; color++) {
        for(int x = 0; x < GAMEMAP_WIDTH; x++) {
            if(gameMap[0][x] != puyogame::Puyo::NONE)
                continue;
            bool isAlive = 1;
            auto newGameMap = this->oneFall(gameMap, x, color, isAlive);
            if(!isAlive)
                continue;
            int r = 0;
            auto _ = this->eraseSimulation(gameMap, newGameMap, r);
            reward = std::max(reward, r);
        }
    }
    return reward;
}

VVI puyogame::State::makePuyoSeqs(int n) {
    std::random_device rnd;
    std::mt19937 mt(rnd());
    std::uniform_int_distribution<> rand(1, PUYO_COLOR);

    VVI ret(n, VI(2));
    for(int i = 0; i < n; i++)
        for(int j = 0; j < 2; j++)
            ret[i][j] = rand(mt);
    return ret;
}