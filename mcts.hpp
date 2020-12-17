#pragma once
#include "PuyoGame.hpp"
#include "define.hpp"
#include "tf_utils.hpp"
#include <memory>

namespace mcts {

class MCTS {
  private:
    TF_Graph *model;
    TF_Session *sess;
    std::vector<TF_Output> inputs;
    std::vector<TF_Output> outputs;
    const std::vector<std::int64_t> input_dims = {1, GAMEMAP_HEIGHT,
                                                  GAMEMAP_WIDTH, CHANNEL_SIZE};

    class Node {
      public:
        puyogame::State state;
        float p, w;
        int r, d, n, turn;
        std::vector<std::unique_ptr<Node>> childNodes;
        float evaluate(const VVI &puyoSeqs, MCTS &parent);
        int nextChildNode();
        Node(puyogame::State state, int d, float p, int r, int turn);
    };

  public:
    MCTS();

    TF_Graph *loadGraph(const char *fileName);

    static std::vector<int> legalActions(VVI field);

    static std::vector<int>
    nodesToScores(const std::vector<std::unique_ptr<Node>> &nodes);

    std::vector<int> randomMcts(puyogame::State state, float temperature);

    std::vector<int> normalMcts(puyogame::State state, const VVI &puyoSeqs,
                                float temperature);

    static std::vector<float> boltzman(std::vector<int> scores,
                                       float temperature);

    static std::vector<float> encode(const puyogame::State &state);

    static std::vector<float> data2Binary(const VVI &stage, VI nowPuyo,
                                          VI nextPuyo, int turn);

    bool prepareSession();

    std::pair<std::vector<float>, float> predict(puyogame::State &state);

    bool close();
};

} // namespace mcts