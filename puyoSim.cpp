#define BOOST_PYTHON_STATIC_LIB
#define BOOST_NUMPY_STATIC_LIB

#include "PuyoGame.hpp"
#include "define.hpp"
#include "mcts.hpp"
#include "tf_utils.hpp"
#include <iostream>
#include <random>
#include <stdio.h>
#include <tensorflow/c/c_api.h>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

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
            puyoGame::State dnextState =
                nextState.next(dnext, puyoSeqs[state.turn + 1], r);
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

using HIST = std::tuple<std::vector<std::vector<int>>, std::vector<std::vector<int>>, std::vector<float>, float>;

void show(int turn, HIST &hist)
{
    auto [gameMap, puyos, policy, r] = hist;
    std::cout << "=== turn :" << turn << " begin ===" << std::endl;
    std::cout << puyos[0][0] << " " << puyos[0][1] << std::endl;
    std::cout << puyos[1][0] << " " << puyos[1][1] << std::endl;
    std::cout << std::endl;
    std::cout << "GAMEMAP : " << std::endl;
    for (int i = 0; i < GAMEMAP_HEIGHT; i++)
    {
        for (int j = 0; j < GAMEMAP_WIDTH; j++)
            std::cout << gameMap[i][j];
        std::cout << std::endl;
    }
    std::cout << "POLICY : ";
    for (int i = 0; i < policy.size(); i++)
    {
        std::cout << policy[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "REWARD : " << r << std::endl;
    std::cout << "=== turn :" << turn << " end ===" << std::endl;
}

int chioceAction(const std::vector<int> &scores)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> distr(scores.begin(), scores.end());
    return distr(gen);
}

std::vector<float> softmax(std::vector<int> &scores)
{
    float sum = 0;
    std::vector<float> ret = std::vector<float>(scores.size());
    for (int i = 0; i < scores.size(); i++)
    {
        sum += scores[i];
    }
    for (int i = 0; i < scores.size(); i++)
    {
        ret[i] = scores[i] / sum;
    }
    return ret;
}

std::vector<HIST> play()
{
    std::vector<HIST> history;
    mcts::MCTS Mcts = mcts::MCTS();

    // モデルの読み込み
    if (Mcts.loadGraph(MODEL_FILENAME) == nullptr)
    {
        std::cout << "Can't load graph" << std::endl;
        return history;
    }
    if (!Mcts.prepareSession())
        return history;

    VVI puyoSeqs = puyoGame::State::makePuyoSeqs(60);
    VVI puyos = puyoGame::State::makePuyoSeqs(2);
    puyoGame::State state =
        puyoGame::State(VVI(GAMEMAP_HEIGHT, VI(GAMEMAP_WIDTH)), puyos, 0);

    std::vector<int> rewards;

    while (1)
    {
        int rensa = state.calcMaxReward();
        if (state.isDone())
            break;
        std::vector<int> scores = Mcts.mctsScores(state, 1.0);
        std::vector<float> softmaxScores = softmax(scores);
        std::vector<int> legalActions = state.legalActions();
        std::vector<float> policies(ACTION_KIND);
        for (int i = 0; i < scores.size(); i++)
            policies[legalActions[i]] = softmaxScores[i];

        int action = legalActions[chioceAction(scores)];

        rewards.push_back(state.calcMaxReward());
        history.emplace_back(state.gameMap, state.puyos, policies, 0);

        int _ = 0;
        state = state.next(action, puyoSeqs[state.turn], _);

        std::cout << "turn" << state.turn << ": "
                  << "rensa. " << rensa << ", action. " << action << std::endl;
    }

    int rMax = 0;
    for (int i = rewards.size() - 1; i >= 0; i--)
    {
        rMax = std::max(rMax, rewards[i]);
        auto &r = std::get<3>(history[i]);
        r = rMax;
    }

    // for (int i = 0; i < history.size(); i++)
    // {
    //     show(i, history[i]);
    // }

    if (!Mcts.closeSession())
        return history;
    return history;
}

namespace python = boost::python;
namespace np = boost::python::numpy;
const int MAX_X = 100;
int main()
{
    Py_Initialize();
    np::initialize();

    auto main_ns = boost::python::import("save").attr("__dict__");

    auto history = play();

    for (int i = 0; i < history.size(); i++)
    {
        show(i, history[i]);
    }

    python::list hist;

    for (auto [gameMap, puyos, policies, value] : history)
    {
        python::list l;

        // gameMap
        boost::python::tuple shapeMap = boost::python::make_tuple(GAMEMAP_HEIGHT, GAMEMAP_WIDTH);
        np::ndarray map = np::zeros(shapeMap, np::dtype::get_builtin<int>());
        for (int i = 0; i < GAMEMAP_HEIGHT; i++)
            for (int j = 0; j < GAMEMAP_WIDTH; j++)
                map[i][j] = gameMap[i][j];
        l.append(map);

        // puyos
        boost::python::tuple shapePuyos = boost::python::make_tuple(2, 2);
        np::ndarray puy = np::zeros(shapeMap, np::dtype::get_builtin<int>());
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                puy[i][j] = puyos[i][j];
        l.append(puy);

        // policies
        boost::python::tuple shapePolicies = boost::python::make_tuple(ACTION_KIND);
        np::ndarray pol = np::zeros(shapePolicies, np::dtype::get_builtin<float>());
        for (int i = 0; i < ACTION_KIND; i++)
            pol[i] = policies[i];
        l.append(pol);

        // value
        l.append(value);

        hist.append(l);
    }

    std::cout
        << "a" << std::endl;
    int a;
    std::cin >> a;

    auto func = main_ns["write_data"];
    auto pyresult_numpy = func(hist);
    std::cin >> a;
}