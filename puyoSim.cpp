#define BOOST_PYTHON_STATIC_LIB
#define BOOST_NUMPY_STATIC_LIB

#include "PuyoGame.hpp"
#include "define.hpp"
#include "mcts.hpp"
#include "tf_utils.hpp"
#include "threadPool.hpp"
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <random>
#include <stdio.h>
#include <tensorflow/c/c_api.h>
#include <time.h>
#define MODEL_FILENAME                                                         \
    "C:/Users/rokahikou/Ohsuga_lab/AlphaPuyo/resources/best.pb"

const std::string MODEL_DIR =
    "C:/Users/rokahikou/Ohsuga_lab/AlphaPuyo/resources";

namespace python = boost::python;
namespace np = boost::python::numpy;

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

using HIST =
    std::tuple<std::vector<std::vector<int>>, std::vector<std::vector<int>>,
               std::vector<float>, float>;

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

int choiceRandomIdx(const std::vector<int> &scores) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> distr(scores.begin(), scores.end());
    return distr(gen);
}

// int choiceMaxIdx(const std::vector<int> &scores) {
//     int ret = 0;
//     int mx = 0;
//     for(int i = 0; i < scores.size(); i++) {
//         if(mx < scores[i]) {
//             ret = i;
//             mx = scores[i];
//         }
//     }
//     return ret;
// }

std::vector<float> softmax(std::vector<int> &scores) {
    float sum = 0;
    std::vector<float> ret = std::vector<float>(scores.size());
    for(int i = 0; i < scores.size(); i++) {
        sum += scores[i];
    }
    for(int i = 0; i < scores.size(); i++) {
        ret[i] = scores[i] / sum;
    }
    return ret;
}

/*--- 現在の時刻を表示する ---*/
void putTime() {
    time_t current;
    struct tm *local;
    time(&current);              /* 現在の時刻を取得 */
    local = localtime(&current); /* 地方時の構造体に変換 */
    printf("%02d:%02d:%02d :: ", local->tm_hour, local->tm_min, local->tm_sec);
}

std::vector<HIST> play(int number) {
    std::vector<HIST> history;
    mcts::MCTS Mcts = mcts::MCTS();

    putTime();
    std::cout << "Play " << number << " started." << std::endl;

    // モデルの読み込み
    if(Mcts.loadGraph(MODEL_FILENAME) == nullptr) {
        throw("Can't load graph.");
    }
    if(!Mcts.prepareSession()) {
        throw("Can't prepare session.");
    }

    VVI puyoSeqs = puyogame::State::makePuyoSeqs(60);
    VVI puyos = puyogame::State::makePuyoSeqs(2);
    puyogame::State state =
        puyogame::State(VVI(GAMEMAP_HEIGHT, VI(GAMEMAP_WIDTH)), puyos, 0);

    std::vector<int> rewards;

    while(1) {
        int rensa = state.calcMaxReward();
        if(state.isDone())
            break;
        if(number % THREAD_NUM == 0) {
            std::cout << "turn : " << state.turn << std::endl;
        }

        // random or normal
        std::vector<int> scores = Mcts.randomMcts(state, 1.0);
        std::vector<float> softmaxScores = softmax(scores);
        std::vector<int> legalActions = state.legalActions();
        std::vector<float> policies(ACTION_KIND);
        for(int i = 0; i < scores.size(); i++)
            policies[legalActions[i]] = softmaxScores[i];

        int action = legalActions[choiceRandomIdx(scores)];

        rewards.push_back(state.calcMaxReward());
        history.emplace_back(state.gameMap, state.puyos, policies, 0);
        // show(state.turn, history.back());

        int _ = 0;
        state = state.next(action, puyoSeqs[state.turn], _);
    }

    int rMax = 0;
    for(int i = rewards.size() - 1; i >= 0; i--) {
        rMax = std::max(rMax, rewards[i]);
        auto &r = std::get<3>(history[i]);
        r = rMax;
    }

    if(!Mcts.close()) {
        throw("Can't close session.");
    }

    putTime();
    std::cout << "Play " << number << " ended.  maxReward : " << rMax
              << std::endl;

    return history;
}

std::pair<int, int> evalPlay(int number) {
    mcts::MCTS Mcts = mcts::MCTS();

    putTime();
    std::cout << "Play " << number << " started." << std::endl;

    // モデルの読み込み
    if(Mcts.loadGraph(MODEL_FILENAME) == nullptr) {
        throw("Can't load graph.");
    }
    if(!Mcts.prepareSession()) {
        throw("Can't prepare session.");
    }

    VVI puyoSeqs = puyogame::State::makePuyoSeqs(60);
    VVI puyos = puyogame::State::makePuyoSeqs(2);
    puyogame::State state =
        puyogame::State(VVI(GAMEMAP_HEIGHT, VI(GAMEMAP_WIDTH)), puyos, 0);

    int reward = 0;
    int potential = 0;
    while(1) {
        int rensa = state.calcMaxReward();
        if(state.isDone())
            break;
        // random or normal
        std::vector<int> scores = Mcts.randomMcts(state, 1.0);
        std::vector<int> legalActions = state.legalActions();

        int action = legalActions[choiceRandomIdx(scores)];

        potential = std::max(reward, state.calcMaxReward());

        int _ = 0;
        state = state.next(action, puyoSeqs[state.turn], _);
        reward = std::max(reward, _);
    }

    if(!Mcts.close()) {
        throw("Can't close session.");
    }

    putTime();
    std::cout << "Evaluate " << number << " ended." << std::endl;

    return std::make_pair(reward, potential);
}

void saveData(std::vector<HIST> &histories) {
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

int main(int argc, char *argv[]) {
    Py_Initialize();
    np::initialize();

    if(argc <= 1) {
        play(0);
    }
    if(strcmp(argv[1], "self") == 0) {
        try {
            std::vector<HIST> histories;
            std::vector<std::future<std::vector<HIST>>> futures;
            for(int i = 0; i < THREAD_NUM; i++) {
                futures.push_back(std::async(std::launch::async, play, i));
            }
            for(auto &f : futures) {
                auto hist = f.get();
                for(auto &h : hist) {
                    histories.emplace_back(h);
                }
            }

            saveData(histories);
        } catch(std::string str) {
            std::cout << str << std::endl;
        }
    } else if(strcmp(argv[1], "eval") == 0) {
        try {
            using pii = std::pair<int, int>;
            int sumr = 0, sump = 0;
            std::vector<std::future<pii>> futures;
            for(int i = 0; i < THREAD_NUM; i++) {
                futures.push_back(
                    std::async(std::launch::async, evalPlay, THREAD_NUM + i));
            }
            for(auto &f : futures) {
                auto rewards = f.get();
                sumr += rewards.first;
                sump += rewards.second;
            }

            std::ofstream output(
                "C:/Users/rokahikou/Ohsuga_lab/AlphaPuyo/evaluate.log",
                std::ios::app);
            output << "Average Rewards : " << sumr / (double)EN_GAME_COUNT
                   << " Average Potentials : " << sump / (double)EN_GAME_COUNT
                   << std::endl;
            output.close();
            std::filesystem::copy(
                MODEL_DIR + "/latest.h5", MODEL_DIR + "/best.h5",
                std::filesystem::copy_options::overwrite_existing);
        } catch(std::string str) {
            std::cout << str << std::endl;
        }
    } else {
        std::cout << "Invalid argument!" << std::endl;
        return 1;
    }
}