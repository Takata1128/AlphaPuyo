#pragma once
#include "PuyoGame.hpp"
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <iostream>
#include <random>
#include <time.h>
#include <vector>
namespace util {

namespace python = boost::python;
namespace np = boost::python::numpy;

using HIST =
    std::tuple<std::vector<std::vector<int>>, std::vector<std::vector<int>>,
               std::vector<float>, float>;

int greedy(puyogame::State state, const VVI &puyoSeqs) {
    int value = 0;
    std::vector<int> actions;
    auto legalActions = state.legalActions();
    for(int next : legalActions) {
        int r;
        puyogame::State nextState = state.next(next, puyoSeqs[state.turn], r);
        auto nextLegalActions = nextState.legalActions();
        for(int dnext : nextLegalActions) {
            puyogame::State dnextState =
                nextState.next(dnext, puyoSeqs[state.turn + 1], r);
            int score = dnextState.calcMaxReward();
            if(score > value) {
                actions.clear();
                actions.push_back(next);
                value = score;
            } else if(score == value) {
                actions.push_back(next);
            }
        }
    }
    std::random_device rnd;
    std::mt19937 mt(rnd());
    if(actions.empty()) {
        std::uniform_int_distribution<> rand(0, legalActions.size() - 1);
        return legalActions[rand(mt)];
    } else {
        std::uniform_int_distribution<> rand(0, actions.size() - 1);
        return actions[rand(mt)];
    }
}

void show(puyogame::State state, const std::vector<float> &policies) {
    std::cout << "=== turn :" << state.turn << " begin ===" << std::endl;
    std::cout << state.puyos[0][0] << " " << state.puyos[0][1] << std::endl;
    std::cout << state.puyos[1][0] << " " << state.puyos[1][1] << std::endl;
    std::cout << std::endl;
    std::cout << "GAMEMAP : " << std::endl;
    for(int i = 0; i < GAMEMAP_HEIGHT; i++) {
        for(int j = 0; j < GAMEMAP_WIDTH; j++)
            std::cout << state.gameMap[i][j];
        std::cout << std::endl;
    }
    std::cout << "POLICY : ";
    for(int i = 0; i < policies.size(); i++) {
        std::cout << policies[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "REWARD : " << state.calcMaxReward() << std::endl;
    std::cout << "=== turn :" << state.turn << " end ===" << std::endl;
}

void show(int turn, HIST &hist) {
    auto [gameMap, puyos, policy, r] = hist;
    std::cout << "=== turn :" << turn << " begin ===" << std::endl;
    std::cout << puyos[0][0] << " " << puyos[0][1] << std::endl;
    std::cout << puyos[1][0] << " " << puyos[1][1] << std::endl;
    std::cout << std::endl;
    std::cout << "GAMEMAP : " << std::endl;
    for(int i = 0; i < GAMEMAP_HEIGHT; i++) {
        for(int j = 0; j < GAMEMAP_WIDTH; j++)
            std::cout << gameMap[i][j];
        std::cout << std::endl;
    }
    std::cout << "POLICY : ";
    for(int i = 0; i < policy.size(); i++) {
        std::cout << policy[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "REWARD : " << r << std::endl;
    std::cout << "=== turn :" << turn << " end ===" << std::endl;
}

int choiceRandomIdx(const std::vector<double> &scores) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<int> distr(scores.begin(), scores.end());
    return distr(gen);
}

/*--- 現在の時刻を表示する ---*/
void putTime() {
    time_t current;
    struct tm *local;
    time(&current);              /* 現在の時刻を取得 */
    local = localtime(&current); /* 地方時の構造体に変換 */
    printf("%02d:%02d:%02d :: ", local->tm_hour, local->tm_min, local->tm_sec);
}

void saveData(std::vector<HIST> &histories) {
    Py_Initialize();
    np::initialize();
    auto main_ns = boost::python::import("save").attr("__dict__");

    python::list hist;

    for(auto [gameMap, puyos, policies, value] : histories) {
        python::list l;

        // gameMap
        boost::python::tuple shapeMap =
            boost::python::make_tuple(GAMEMAP_HEIGHT, GAMEMAP_WIDTH);
        np::ndarray map = np::zeros(shapeMap, np::dtype::get_builtin<int>());
        for(int i = 0; i < GAMEMAP_HEIGHT; i++)
            for(int j = 0; j < GAMEMAP_WIDTH; j++)
                map[i][j] = gameMap[i][j];
        l.append(map);

        // puyos
        boost::python::tuple shapePuyos = boost::python::make_tuple(2, 2);
        np::ndarray puy = np::zeros(shapeMap, np::dtype::get_builtin<int>());
        for(int i = 0; i < 2; i++)
            for(int j = 0; j < 2; j++)
                puy[i][j] = puyos[i][j];
        l.append(puy);

        // policies
        boost::python::tuple shapePolicies =
            boost::python::make_tuple(ACTION_KIND);
        np::ndarray pol =
            np::zeros(shapePolicies, np::dtype::get_builtin<float>());
        for(int i = 0; i < ACTION_KIND; i++)
            pol[i] = policies[i];
        l.append(pol);

        // value
        l.append(value);
        hist.append(l);
    }

    auto func = main_ns["write_data"];
    auto pyresult_numpy = func(hist);
}
} // namespace util