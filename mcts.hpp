#include "PuyoGame.hpp"
#include "define.hpp"
#include "tf_utils.hpp"
#include <algorithm>
#include <numeric>
namespace mcts {
class Node {
  public:
    State state;
    float p, r;
    int d, w, n, turn;
    std::vector<Node> childNodes;
    Node(State state, int d, float p, float r, int turn);
    float evaluate();
    Node nextChildNode();
};

std::vector<Puyo> makePuyoList(VVI next2);

std::vector<int> legalActions(VVI field);

std::vector<int> nodesToScores(const std::vector<Node> &nodes);

std::vector<float> mctsScores(TF_Graph *graph, State state, float temperature);

std::vector<float> encode(const State &state);

std::vector<float> data2Binary(const VVI &stage, VI nowPuyo, VI nextPuyo,
                               int turn);

std::vector<float> predict(TF_Graph *graph, const State &state);

} // namespace mcts