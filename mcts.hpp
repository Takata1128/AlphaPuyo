#pragma once
#include "PuyoGame.hpp"
#include "define.hpp"
#include "tf_utils.hpp"
#include <memory>

namespace mcts
{

  class Node
  {
  private:
    TF_Graph *model;
    TF_Session *sess;

  public:
    puyoGame::State state;
    float p, w;
    int r, d, n, turn;
    std::vector<std::unique_ptr<Node>> childNodes;
    Node(puyoGame::State state, int d, float p, int r, int turn, TF_Graph *graph, TF_Session *sess);
    float evaluate(const VVI &puyoSeqs);
    int nextChildNode();
  };

  class MCTS
  {
  private:
    TF_Graph *model;
    TF_Session *sess;

  public:
    MCTS()
    {
      model = nullptr;
      sess = nullptr;
    }

    TF_Graph *loadGraph(const char *fileName);

    static std::vector<int> legalActions(VVI field);

    static std::vector<int> nodesToScores(const std::vector<std::unique_ptr<mcts::Node>> &nodes);

    std::vector<int> mctsScores(puyoGame::State state, float temperature);

    static std::vector<float> boltzman(std::vector<int> scores, float temperature);

    static std::vector<float> encode(const puyoGame::State &state);

    static std::vector<float> data2Binary(const VVI &stage, VI nowPuyo, VI nextPuyo,
                                          int turn);

    bool prepareSession();

    static std::vector<float> predict(const puyoGame::State &state, TF_Graph *model, TF_Session *sess);

    bool closeSession();
  };

} // namespace mcts