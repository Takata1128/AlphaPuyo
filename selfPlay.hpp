#pragma once
#include "PuyoGame.hpp"
#include "define.hpp"
#include "mcts.hpp"
#include "util.hpp"
#include <algorithm>
#include <iostream>
#include <tuple>
#include <vector>

#define MODEL_FILENAME                                                         \
    "C:/Users/rokahikou/Ohsuga_lab/AlphaPuyo/resources/best.pb"
#define EVAL_MODEL_FILENAME                                                    \
    "C:/Users/rokahikou/Ohsuga_lab/AlphaPuyo/resources/best.pb"

namespace selfplay {
using HIST =
    std::tuple<std::vector<std::vector<int>>, std::vector<std::vector<int>>,
               std::vector<float>, float>;

std::vector<HIST> play(int number, bool isDebug = false) {
    std::vector<HIST> history;
    mcts::MCTS Mcts = mcts::MCTS();

    // モデルの読み込み
    if(Mcts.loadGraph(MODEL_FILENAME) == nullptr) {
        throw("Can't load graph.");
    }
    if(!Mcts.prepareSession()) {
        throw("Can't prepare session.");
    }

    for(int k = 0; k < SP_GAME_COUNT / THREAD_NUM; k++) {
        util::putTime();
        std::cout << "Play " << number * (SP_GAME_COUNT / THREAD_NUM) + k
                  << " started." << std::endl;
        VVI puyoSeqs = puyogame::State::makePuyoSeqs(MAX_STEP + 10);
        VVI puyos = puyogame::State::makePuyoSeqs(2);
        puyogame::State state =
            puyogame::State(VVI(GAMEMAP_HEIGHT, VI(GAMEMAP_WIDTH)), puyos, 0);
        std::vector<int> rewards;
        std::vector<HIST> tmpHistory;
        while(1) {
            int rensa = state.calcMaxReward();
            if(state.isDone())
                break;
            std::vector<double> scores =
                Mcts.mcts(state, puyoSeqs, MCTS_TEMPERATURE, MCTS_TYPE);
            std::vector<int> legalActions = state.legalActions();
            std::vector<float> policies(ACTION_KIND);
            for(int i = 0; i < scores.size(); i++)
                policies[legalActions[i]] = scores[i];
            int action = legalActions[util::choiceRandomIdx(scores)];
            rewards.push_back(state.calcMaxReward());
            tmpHistory.emplace_back(state.gameMap, state.puyos, policies, 0);
            if(isDebug)
                util::show(state, policies);
            int _ = 0;
            state = state.next(action, puyoSeqs[state.turn], _);
            if(isDebug)
                std::cout << "rewards : " << _ << std::endl;
        }

        int rMax = 0;
        for(int i = rewards.size() - 1; i >= 0; i--) {
            rMax = std::max(rMax, rewards[i]);
            auto &r = std::get<3>(tmpHistory[i]);
            r = rMax;
        }

        for(auto h : tmpHistory)
            history.emplace_back(h);

        util::putTime();
        std::cout << "Play " << number * (SP_GAME_COUNT / THREAD_NUM) + k
                  << " ended.  maxPotential : " << rMax
                  << ", turn: " << state.turn << std::endl;
        // if(isDebug)
        //     break;
    }

    if(!Mcts.close()) {
        throw("Can't close session.");
    }
    return history;
}

std::tuple<std::vector<HIST>, std::vector<int>, std::vector<int>>
evalPlay(int number) {
    std::vector<HIST> history;
    std::vector<int> rewardsAll, potentialsAll;
    mcts::MCTS Mcts = mcts::MCTS();

    // モデルの読み込み
    if(Mcts.loadGraph(MODEL_FILENAME) == nullptr) {
        throw("Can't load graph.");
    }
    if(!Mcts.prepareSession()) {
        throw("Can't prepare session.");
    }
    for(int k = 0; k < EN_GAME_COUNT / THREAD_NUM; k++) {
        util::putTime();
        VVI puyoSeqs = puyogame::State::makePuyoSeqs(MAX_STEP + 10);
        VVI puyos = puyogame::State::makePuyoSeqs(2);
        puyogame::State state =
            puyogame::State(VVI(GAMEMAP_HEIGHT, VI(GAMEMAP_WIDTH)), puyos, 0);
        std::vector<int> rewards;
        std::vector<HIST> tmpHistory;
        int potential = 0, reward = 0;
        while(1) {
            if(state.isDone())
                break;
            std::vector<double> scores =
                Mcts.mcts(state, puyoSeqs, MCTS_TEMPERATURE, MCTS_TYPE);
            std::vector<int> legalActions = state.legalActions();
            std::vector<float> policies(ACTION_KIND);
            for(int i = 0; i < scores.size(); i++)
                policies[legalActions[i]] = scores[i];
            int actionIdx = std::distance(
                scores.begin(), std::max_element(scores.begin(), scores.end()));
            int action = legalActions[actionIdx];
            int rensa = state.calcMaxReward();
            rewards.push_back(rensa);
            tmpHistory.emplace_back(state.gameMap, state.puyos, policies, 0);
            int _ = 0;
            util::show(state.turn, tmpHistory.back());
            state = state.next(action, puyoSeqs[state.turn], _);
            reward = std::max(_, reward);
            potential = std::max(rensa, potential);
        }

        int rMax = 0;
        for(int i = rewards.size() - 1; i >= 0; i--) {
            rMax = std::max(rMax, rewards[i]);
            auto &r = std::get<3>(tmpHistory[i]);
            r = rMax;
        }

        for(auto h : tmpHistory)
            history.emplace_back(h);
        rewardsAll.push_back(reward);
        potentialsAll.push_back(potential);

        util::putTime();
        std::cout << "Evaluate " << number * (EN_GAME_COUNT / THREAD_NUM) + k
                  << " ended.  maxPotential : " << potential
                  << ", turn: " << state.turn << std::endl;
    }

    if(!Mcts.close()) {
        throw("Can't close session.");
    }

    return std::make_tuple(history, rewardsAll, potentialsAll);
}

std::tuple<std::vector<HIST>, std::vector<int>, std::vector<int>>
greedyPlay(int number) {
    std::vector<HIST> history;
    std::vector<int> rewardsAll, potentialsAll;
    mcts::MCTS Mcts = mcts::MCTS();

    // モデルの読み込み
    if(Mcts.loadGraph(MODEL_FILENAME) == nullptr) {
        throw("Can't load graph.");
    }
    if(!Mcts.prepareSession()) {
        throw("Can't prepare session.");
    }
    for(int k = 0; k < EN_GAME_COUNT; k++) {
        util::putTime();
        std::cout << "Evaluate " << number * (EN_GAME_COUNT / THREAD_NUM) + k
                  << " started." << std::endl;
        VVI puyoSeqs = puyogame::State::makePuyoSeqs(MAX_STEP + 10);
        VVI puyos = puyogame::State::makePuyoSeqs(2);
        puyogame::State state =
            puyogame::State(VVI(GAMEMAP_HEIGHT, VI(GAMEMAP_WIDTH)), puyos, 0);
        std::vector<int> rewards;
        std::vector<HIST> tmpHistory;
        int potential = 0, reward = 0;
        while(1) {
            if(state.isDone())
                break;
            int action = util::greedy(state, puyoSeqs);
            int rensa = state.calcMaxReward();
            rewards.push_back(rensa);
            int _ = 0;
            state = state.next(action, puyoSeqs[state.turn], _);
            reward = std::max(_, reward);
            potential = std::max(rensa, potential);
        }

        for(auto h : tmpHistory)
            history.emplace_back(h);
        rewardsAll.push_back(reward);
        potentialsAll.push_back(potential);

        util::putTime();
        std::cout << "Evaluate " << number * (EN_GAME_COUNT / THREAD_NUM) + k
                  << " ended.  maxPotential : " << potential
                  << ", turn: " << state.turn << std::endl;
    }

    if(!Mcts.close()) {
        throw("Can't close session.");
    }

    return std::make_tuple(history, rewardsAll, potentialsAll);
}

} // namespace selfplay