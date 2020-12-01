#include "define.hpp"
#include "tf_utils.hpp"
#include "PuyoGame.hpp"
#include "mcts.hpp"
#include <iostream>
#include <stdio.h>
#include <random>
#include <tensorflow/c/c_api.h>

#define MODEL_FILENAME \
    "C:/Users/s.takata/Documents/tensorflow_cpp/build/resources/best.pb"

int greedy(puyoGame::State state, const VVI &puyoSeqs)
{
    int value = 0;
    std::vector<int> actions;
    auto legalActions = state.legalActions();
    for (int next : legalActions)
    {
        int r;
        puyoGame::State nextState = state.next(next, puyoSeqs[state.turn], r);
        auto nextLegalActions = nextState.legalActions();
        for (int dnext : nextLegalActions)
        {
            puyoGame::State dnextState = nextState.next(dnext, puyoSeqs[state.turn + 1], r);
            int score = dnextState.calcMaxReward();
            if (score > value)
            {
                actions.clear();
                actions.push_back(next);
                value = score;
            }
            else if (score == value)
            {
                actions.push_back(next);
            }
        }
    }
    std::random_device rnd;
    std::mt19937 mt(rnd());
    if (actions.empty())
    {
        std::uniform_int_distribution<> rand(0, legalActions.size() - 1);
        return legalActions[rand(mt)];
    }
    else
    {
        std::uniform_int_distribution<> rand(0, actions.size() - 1);
        return actions[rand(mt)];
    }
}

void show(puyoGame::State state)
{
    std::cout << "=== turn :" << state.turn << " begin ===" << std::endl;
    std::cout << state.puyos[0][0] << " " << state.puyos[0][1] << std::endl;
    std::cout << state.puyos[1][0] << " " << state.puyos[1][1] << std::endl;
    std::cout << std::endl;

    for (int i = 0; i < GAMEMAP_HEIGHT; i++)
    {
        for (int j = 0; j < GAMEMAP_WIDTH; j++)
            std::cout << state.gameMap[i][j];
        std::cout << std::endl;
    }
    std::cout << "=== turn :" << state.turn << " end ===" << std::endl;
}

int chioceAction(const std::vector<int> &scores)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> distr(scores.begin(), scores.end());
    return distr(gen);
}

int main()
{
    // モデルの読み込み
    mcts::loadGraph(MODEL_FILENAME);
    if (mcts::getModel() == nullptr)
    {
        std::cout << "Can't load graph" << std::endl;
        return 1;
    }

    VVI puyoSeqs = mcts::makePuyoSeqs(60);
    VVI puyos(2, VI(2));
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            puyos[i][j] = rand() % 4 + 1;
    puyoGame::State state = puyoGame::State(VVI(GAMEMAP_HEIGHT, VI(GAMEMAP_WIDTH)), puyos, 0);
    while (1)
    {
        int rensa = state.calcMaxReward();
        std::cout << "rensa : " << rensa << std::endl;
        show(state);
        if (state.isDone())
            break;
        std::vector<int> scores = mcts::mctsScores(state, 1.0);
        int action = chioceAction(scores);
        std::cout << "action : " << action << std::endl;
        int reward = 0;
        state = state.next(action, puyoSeqs[state.turn], reward);
        int a;
        std::cin >> a;
    }
    return 0;
}