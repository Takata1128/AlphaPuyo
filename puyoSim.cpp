#define BOOST_PYTHON_STATIC_LIB
#define BOOST_NUMPY_STATIC_LIB

#include "define.hpp"
#include "mcts.hpp"
#include "selfPlay.hpp"
#include "tf_utils.hpp"
#include "threadPool.hpp"
#include "util.hpp"
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <stdio.h>
#include <tensorflow/c/c_api.h>
#include <vector>

using HIST =
    std::tuple<std::vector<std::vector<int>>, std::vector<std::vector<int>>,
               std::vector<float>, float>;

int main(int argc, char *argv[]) {
    if(argc <= 1) {
        selfplay::play(0);
    }
    if(strcmp(argv[1], "self") == 0) {
        try {
            std::vector<HIST> histories;
            std::vector<std::future<std::vector<HIST>>> futures;
            for(int i = 0; i < THREAD_NUM; i++) {
                futures.push_back(
                    std::async(std::launch::async, selfplay::play, i));
            }
            for(auto &f : futures) {
                auto hist = f.get();
                for(auto h : hist) {
                    histories.emplace_back(h);
                }
            }
            util::saveData(histories);
        } catch(std::string str) {
            std::cout << str << std::endl;
        }
    } else if(strcmp(argv[1], "eval") == 0) {
        try {
            int sumr = 0, sump = 0;
            std::vector<HIST> histories;
            std::vector<int> rewardsAll, potentialsAll;
            std::vector<std::future<std::tuple<
                std::vector<HIST>, std::vector<int>, std::vector<int>>>>
                futures;
            for(int i = 0; i < THREAD_NUM; i++) {
                futures.push_back(
                    std::async(std::launch::async, selfplay::evalPlay, i));
            }
            for(auto &f : futures) {
                auto data = f.get();
                auto [hist, rewds, pots] = data;
                for(auto h : hist) {
                    histories.emplace_back(h);
                }
                for(auto r : rewds) {
                    rewardsAll.emplace_back(r);
                }
                for(auto p : pots) {
                    potentialsAll.emplace_back(p);
                }
            }

            std::ofstream output(
                "C:/Users/rokahikou/Ohsuga_lab/AlphaPuyo/evaluate.log",
                std::ios::app);
            for(int i = 0; i < rewardsAll.size(); i++) {
                output << rewardsAll[i]
                       << (i == rewardsAll.size() - 1 ? "/n" : ",");
            }
            for(int i = 0; i < potentialsAll.size(); i++) {
                output << potentialsAll[i]
                       << (i == potentialsAll.size() - 1 ? "/n" : ",");
            }
            output.close();
            util::saveData(histories);
        } catch(std::string str) {
            std::cout << str << std::endl;
        }
    } else {
        std::cout << "Invalid argument!" << std::endl;
        return 1;
    }
}