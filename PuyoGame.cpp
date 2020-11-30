#include "puyoGame.hpp"
#include <cassert>
#include <queue>

const int dx[] = {0, -1, 0, 1};
const int dy[] = {1, 0, -1, 0};

Puyo::Puyo(int color1, int color2)
{
    this->direction = UP;
    this->color1 = color1;
    this->color2 = color2;
}

void Puyo::set_direct(int direct) { this->direction = direct; }

State::State()
{
    this->gameMap = VVI();
    this->puyos = VVI(2, VI(2));
    this->turn = 0;
}

State::State(VVI gameMap, VVI puyos, int turn)
{
    this->gameMap = gameMap;
    this->puyos = puyos;
    this->turn = turn;
}

bool State::isDone() { return this->isEnd() || this->isLose(); }

bool State::isLose() { return this->gameMap[1][2] != Puyo::NONE; }

bool State::isEnd() { return this->turn == MAX_STEP; }

Puyo State::getNextPuyo() { return Puyo(puyos[0][0], puyos[0][1]); }

inline int getFallY(const VVI &stage, int x)
{
    size_t fallY = 0;
    while (fallY + 1 < GAMEMAP_HEIGHT && stage[fallY + 1][x] == Puyo::NONE)
        fallY++;
    return fallY;
}

VVI State::oneFall(VVI stage, size_t x, int color, bool &isAlive)
{
    assert(stage[0][x] == Puyo::NONE);

    size_t fallY = getFallY(stage, x);
    stage[fallY][x] = color;
    isAlive = (stage[1][2] == Puyo::NONE);
    return stage;
}

// 組ぷよを落とす
VVI State::puyoFall(VVI stage, size_t x, Puyo puyo, bool &isAlive)
{
    if (puyo.direction == Puyo::UP)
    {
        assert(stage[0][x] == Puyo::NONE && stage[1][x] == Puyo::NONE);
        size_t fallY = getFallY(stage, x);
        stage[fallY][x] = puyo.color1;
        fallY = getFallY(stage, x);
        stage[fallY][x] = puyo.color2;
    }
    else if (puyo.direction == Puyo::RIGHT)
    {
        assert(x + 1 < GAMEMAP_WIDTH);
        assert(stage[0][x] == Puyo::NONE && stage[0][x + 1] == Puyo::NONE);
        size_t fallY = getFallY(stage, x);
        stage[fallY][x] = puyo.color1;
        fallY = getFallY(stage, x + 1);
        stage[fallY][x + 1] = puyo.color2;
    }
    else if (puyo.direction == Puyo::DOWN)
    {
        assert(stage[0][x] == Puyo::NONE && stage[1][x] == Puyo::NONE);
        size_t fallY = getFallY(stage, x);
        stage[fallY][x] = puyo.color2;
        fallY = getFallY(stage, x);
        stage[fallY][x] = puyo.color1;
    }
    else if (puyo.direction == Puyo::LEFT)
    {
        assert(x - 1 >= 0);
        assert(stage[0][x] == Puyo::NONE && stage[0][x - 1] == Puyo::NONE);
        size_t fallY = getFallY(stage, x);
        stage[fallY][x] = puyo.color1;
        fallY = getFallY(stage, x - 1);
        stage[fallY][x - 1] = puyo.color2;
    }
    isAlive = (stage[1][2] == Puyo::NONE);
    return stage;
}

VVI State::erase(VVI stage, VVI newStage, int &getScore)
{
    getScore = 0;
    std::vector<std::pair<int, int>> vec;
    for (int i = 0; i < GAMEMAP_HEIGHT; i++)
    {
        for (int j = 0; j < GAMEMAP_WIDTH; j++)
        {
            if (stage[i][j] - newStage[i][j] != 0)
            {
                vec.emplace_back(j, i);
            }
        }
    }
    for (auto p : vec)
    {
        int x = p.first, y = p.second;
        int color = newStage[y][x];
        int counter = 0;
        VVI cpStage = this->erasePuyo(newStage, x, y, color, counter);
        if (counter >= 4)
        {
            getScore += counter * 5;
            newStage = cpStage;
        }
    }
    return newStage;
}

VVI State::erasePuyo(VVI stage, size_t sx, size_t sy, int color, int &counter)
{
    counter = 0;
    if (color == Puyo::OJAMA)
    {
        return stage;
    }
    if (color == Puyo::NONE)
    {
        return stage;
    }
    std::queue<std::pair<int, int>> que;
    que.emplace(sx, sy);
    while (!que.empty())
    {
        auto [x, y] = que.front();
        que.pop();
        if (stage[y][x] == Puyo::OJAMA)
        {
            stage[y][x] = Puyo::NONE;
            continue;
        }
        counter++;
        stage[y][x] = Puyo::NONE;
        for (int i = 0; i < 4; i++)
        {
            int nx = x + dx[i], ny = y + dy[i];
            if (nx >= 0 && nx < GAMEMAP_WIDTH && ny >= 0 &&
                ny < GAMEMAP_HEIGHT)
            {
                if (stage[ny][nx] == Puyo::OJAMA || stage[ny][nx] == color)
                {
                    que.emplace(nx, ny);
                }
            }
        }
    }
    return stage;
}

VVI State::eraseSimulation(VVI gameMap, VVI newGameMap, int &reward)
{
    int counter = 0;
    reward = 0;
    while (1)
    {
        // 消す
        int getScore = 0;
        VVI newStage = this->erase(gameMap, newGameMap, getScore);
        if (getScore > 0)
        {
            counter++;
            reward++;
        }
        else if (getScore == 0)
        {
            gameMap = newStage;
            break;
        }

        // 落とす
        newGameMap = this->fall(newStage);
        gameMap = newStage;
    }
    return gameMap;
}

VVI State::fall(VVI stage)
{
    for (int x = 0; x < GAMEMAP_WIDTH; x++)
    {
        std::queue<int> que;
        for (int y = GAMEMAP_HEIGHT - 1; y >= 0; y--)
        {
            if (stage[y][x] != Puyo::NONE)
            {
                que.push(stage[y][x]);
            }
        }
        for (int y = GAMEMAP_HEIGHT - 1; y >= 0; y--)
        {
            if (que.empty())
            {
                stage[y][x] = Puyo::NONE;
            }
            else
            {
                stage[y][x] = que.front();
                que.pop();
            }
        }
    }
    return stage;
}

VI State::legalActions()
{
    VI ret;
    int actionIdx = 0;
    for (auto action_vec : ACTIONSDICT)
    {
        bool ok = true;
        for (int x = 0; x < GAMEMAP_WIDTH; x++)
        {
            int cnt = 0;
            for (int y = 0; y < GAMEMAP_HEIGHT; y++)
            {
                if (this->gameMap[y][x] == Puyo::NONE)
                    cnt++;
            }
            if (cnt < action_vec[x])
                ok = false;
        }
        if (ok)
            ret.push_back(actionIdx);
        actionIdx++;
    }
    return ret;
}

inline VVI makePuyos(const VI &puyo1, const VI &puyo2)
{
    VVI ret = VVI(2, VI(2));
    ret[0][0] = puyo1[0];
    ret[0][1] = puyo1[1];
    ret[1][0] = puyo2[0];
    ret[1][1] = puyo2[1];
    return ret;
}

// 次状態に遷移
State State::next(int action, VI nextPuyoColors, int &reward)
{
    VVI gameMap = this->gameMap;
    Puyo puyo = getNextPuyo();
    int x;
    if (action < 3)
    {
        x = 0;
        if (action == 0)
            puyo.direction = puyo.RIGHT;
        if (action == 1)
            puyo.direction = puyo.DOWN;
        if (action == 2)
            puyo.direction = puyo.UP;
    }
    else if (action > 18)
    {
        x = 5;
        if (action == 19)
            puyo.direction = puyo.LEFT;
        if (action == 20)
            puyo.direction = puyo.DOWN;
        if (action == 21)
            puyo.direction = puyo.UP;
    }
    else
    {
        puyo.direction = ((action + 1) - 4) % 4 + 1;
        x = ((action - 3) / 4) + 1;
    }

    bool isAlive = 1;
    VVI newGameMap = this->puyoFall(gameMap, x, puyo, isAlive);
    VVI resMap = this->eraseSimulation(gameMap, newGameMap, reward);
    VVI nextPuyos = makePuyos(puyos[1], nextPuyoColors);
    return State(resMap, nextPuyos, this->turn + 1);
}

int State::calcMaxReward()
{
    auto gameMap = this->gameMap;
    int reward = 0;
    for (int color = 1; color <= PUYO_COLOR; color++)
    {
        for (int x = 0; x < GAMEMAP_WIDTH; x++)
        {
            if (gameMap[0][x] != Puyo::NONE)
                continue;
            bool isAlive = 1;
            auto newGameMap = this->oneFall(gameMap, x, color, isAlive);
            if (!isAlive)
                continue;
            int r = 0;
            auto _ = this->eraseSimulation(gameMap, newGameMap, r);
            reward = std::max(reward, r);
        }
    }
    return reward;
}