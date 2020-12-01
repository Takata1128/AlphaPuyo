#pragma once
#include "PuyoGame.hpp"
#include "define.hpp"
#include "tf_utils.hpp"
#include <memory>

namespace mcts
{

  static TF_Graph *model;

  class Node
  {
  public:
    puyoGame::State state;
    float p, w;
    int r, d, n, turn;
    std::vector<std::unique_ptr<Node>> childNodes;
    Node(puyoGame::State state, int d, float p, int r, int turn);
    float evaluate(const VVI &puyoSeqs);
    int nextChildNode();
  };

  void loadGraph(const char *fileName);

  TF_Graph *getModel();

  VVI makePuyoSeqs(const int len);

  std::vector<int> legalActions(VVI field);

  std::vector<int> nodesToScores(const std::vector<std::unique_ptr<mcts::Node>> &nodes);

  std::vector<int> mctsScores(puyoGame::State state, float temperature);

  std::vector<float> boltzman(std::vector<int> scores, float temperature);

  std::vector<float> encode(const puyoGame::State &state);

  std::vector<float> data2Binary(const VVI &stage, VI nowPuyo, VI nextPuyo,
                                 int turn);

  std::vector<float> predict(const puyoGame::State &state);

} // namespace mcts