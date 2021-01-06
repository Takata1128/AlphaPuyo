#pragma once
#include "PuyoGame.hpp"
#include "define.hpp"
#include "tf_utils.hpp"
#include <memory>

namespace mcts {

class MCTS {
  private:
    TF_Graph *mModel;
    TF_Session *mSess;
    TF_Status *mStatus;
    std::vector<TF_Output> mInputs;
    std::vector<TF_Output> mOutputs;
    const std::vector<std::int64_t> mInput_dims = {1, GAMEMAP_HEIGHT,
                                                   GAMEMAP_WIDTH, CHANNEL_SIZE};

    class Node {
      public:
        puyogame::State state;
        float p, w;
        int r, d, n;
        std::vector<Node> childNodes;

        /// @brief evaluate value of node
        /// @param puyoSeqs next puyo sequences
        /// @param parent mcts instance
        /// @return value of node
        float evaluate(const VVI &puyoSeqs, MCTS &parent);

        /// @brief get best child node index
        /// @return best child node index
        int nextChildNode();
        Node(puyogame::State state, int d, float p, int r);
    };

  public:
    MCTS();

    /// @brief load nn model
    /// @return pointer of nn model
    TF_Graph *loadGraph(const char *fileName);

    /// @brief get distributions of next actions from mcts result
    /// @param nodes mcts result nodes
    /// @return distributions of next actions
    static std::vector<int> nodesToScores(const std::vector<Node> &nodes);

    /// @brief
    std::vector<double> randomMcts(puyogame::State state, float temperature);

    std::vector<double> normalMcts(puyogame::State state, const VVI &puyoSeqs,
                                   float temperature);

    /// @brief monte calro tree search
    /// @param temperature temperature of boltzman distribution
    /// @param type SINGLE = 0,RANDOM = 1
    /// @return probability distributions of next actions
    std::vector<double> mcts(puyogame::State state, const VVI &puyoSeqs,
                             float temperature, int type);

    /// @brief convert to boltzman distribution
    /// @return boltzman distribution
    static std::vector<double> boltzman(const std::vector<int> &scores,
                                        const float temperature);

    /// @brief prepare tensorflow session. need to call this before mcts.
    /// @return preparation success or not
    bool prepareSession();

    /// @brief predict policies and value
    /// @param state current state
    /// @return a pair of policies and value
    std::pair<std::vector<float>, float> predict(puyogame::State &state);

    /// @brief close session and delete model
    /// @return close success or not
    bool close();
};

} // namespace mcts