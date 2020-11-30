#include "PuyoGame.hpp"
#include "define.hpp"
#include "tf_utils.hpp"
#include <algorithm>
#include <numeric>
namespace mcts
{

  TF_Graph *graph = nullptr;

  class Node
  {
  public:
    State state;
    float p;
    int r, d, w, n, turn;
    std::vector<Node> childNodes;
    Node(State state, int d, float p, int r, int turn);
    float evaluate(const VVI &puyoSeqs);
    Node nextChildNode();
  };

  VVI makePuyoSeqs(const int len);

  std::vector<int> legalActions(VVI field);

  std::vector<int> nodesToScores(const std::vector<Node> &nodes);

  std::vector<float> mctsScores(TF_Graph *graph, State state, float temperature);

  std::vector<float> encode(const State &state);

  std::vector<float> data2Binary(const VVI &stage, VI nowPuyo, VI nextPuyo,
                                 int turn);

  std::vector<float> predict(TF_Graph *graph, const State &state);

} // namespace mcts