#include "mcts.hpp"
#include "PuyoGame.hpp"
#include "define.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

mcts::Node::Node(puyoGame::State state, int d, float p, int r, int turn)
{
    this->state = state;
    this->p = p;
    this->d = d;
    this->r = r;
    this->w = 0;
    this->n = 0;
    this->turn = turn;
}

float mcts::Node::evaluate(const VVI &puyoSeqs)
{
    if (this->state.isDone() || this->turn == MAX_STEP)
    {
        float value = (this->state.isLose() ? -1 : r);
        this->w += value;
        this->n++;
        return value;
    }

    if (d == EVALUATE_DEPTH)
    {
        float value = this->state.calcMaxReward();
        this->w += value;
        this->n++;
        return value;
    }

    if (this->childNodes.empty())
    {
        std::vector<float> policies = mcts::predict(this->state);
        float value = policies.back();
        std::cout << "predict value: " << value << std::endl;
        policies.pop_back();
        std::cout << "predict policies: ";
        for (auto p : policies)
            std::cout << p << " ";
        std::cout << std::endl;
        this->w += value;
        this->n++;
        VI actions = this->state.legalActions();
        for (int i = 0; i < actions.size(); i++)
        {
            int reward = 0;
            puyoGame::State nextState = this->state.next(actions[i], puyoSeqs[this->d], reward);
            this->childNodes.push_back(std::unique_ptr<Node>(new Node(nextState, this->d + 1, policies[actions[i]], std::max(this->r, reward), this->turn + 1)));
        }
        return value;
    }
    else
    {
        float value = this->childNodes[nextChildNode()]->evaluate(puyoSeqs);
        this->w += value;
        this->n++;
        return value;
    }
}

int mcts::Node::nextChildNode()
{
    auto v = nodesToScores(childNodes);
    int t = std::accumulate(v.begin(), v.end(), 0);
    std::vector<float> pucbValues;
    for (auto &c : childNodes)
    {
        pucbValues.push_back((c->n ? (c->w / (float)c->n) : 0.0) +
                             (w / (float)n) * c->p * sqrt(t) / (float)(1 + c->n));
    }
    std::cout << "w : " << w << std::endl;
    std::cout << "=== pucbValues ===" << std::endl;
    for (int i = 0; i < pucbValues.size(); i++)
    {
        std::cout << pucbValues[i] << " ";
    }
    std::cout << std::endl;

    int idx = std::distance(
        pucbValues.begin(),
        std::max_element(pucbValues.begin(), pucbValues.end()));
    return idx;
}

void mcts::loadGraph(const char *fileName)
{
    model = tf_utils::LoadGraph(fileName);
}

TF_Graph *mcts::getModel()
{
    return model;
}

std::vector<int>
mcts::nodesToScores(const std::vector<std::unique_ptr<mcts::Node>> &nodes)
{
    std::vector<int> ret;
    for (auto &c : nodes)
        ret.push_back(c->n);
    return ret;
}

VVI mcts::makePuyoSeqs(const int len)
{
    VVI ret(len, VI(2));
    for (int i = 0; i < len; i++)
        for (int j = 0; j < 2; j++)
            ret[i][j] = rand() % 4 + 1;
    return ret;
}

std::vector<int> mcts::mctsScores(puyoGame::State state,
                                  float temperature)
{
    int childNodeNum = state.legalActions().size();
    std::vector<int> childNodesPlayCounts(childNodeNum);
    for (int i = 0; i < TRY_COUNT; i++)
    {
        std::cout << "TRY " << i << std::endl;
        VVI puyoSeqs = mcts::makePuyoSeqs(10);
        puyoGame::State tmp = state;
        Node rootNode = Node(tmp, 0, 0, 0, tmp.turn);
        for (int j = 0; j < PV_EVALUATE_COUNT; j++)
        {
            rootNode.evaluate(puyoSeqs);
            std::cout << "evaluate " << j << std::endl;
        }
        for (int j = 0; j < childNodeNum; j++)
        {
            childNodesPlayCounts[j] += rootNode.childNodes[j]->n;
        }
    }
    return childNodesPlayCounts;
}

std::vector<float> mcts::encode(const puyoGame::State &state)
{
    return data2Binary(state.gameMap, state.puyos[0], state.puyos[1],
                       state.turn);
}

std::vector<float> mcts::data2Binary(const VVI &stage, VI nowPuyo, VI nextPuyo,
                                     int turn)
{
    std::vector<std::vector<std::vector<int>>> mat(
        GAMEMAP_HEIGHT, std::vector<std::vector<int>>(
                            GAMEMAP_WIDTH, std::vector<int>(CHANNEL_SIZE)));
    for (int i = 0; i < GAMEMAP_HEIGHT; i++)
        for (int j = 0; j < GAMEMAP_WIDTH; j++)
            mat[i][j][stage[i][j]] = 1;

    VI puyoData = VI();
    for (int i = 0; i < 2; i++)
        puyoData.push_back(nowPuyo[i]);
    for (int i = 0; i < 2; i++)
        puyoData.push_back(nextPuyo[i]);
    for (int k = 0; k < 4; k++)
        for (int i = 0; i < GAMEMAP_HEIGHT; i++)
            for (int j = 0; j < GAMEMAP_WIDTH; j++)
                mat[i][j][puyoData[k] + PUYO_COLOR + k * PUYO_COLOR];
    std::vector<float> ret(GAMEMAP_HEIGHT * GAMEMAP_WIDTH * CHANNEL_SIZE);
    int cur = 0;
    for (int i = 0; i < GAMEMAP_HEIGHT; i++)
        for (int j = 0; j < GAMEMAP_WIDTH; j++)
            for (int k = 0; k < CHANNEL_SIZE; k++)
                ret[cur++] = mat[i][j][k];
    return ret;
}

std::vector<float> mcts::predict(const puyoGame::State &state)
{
    /* prepare input tensor */
    TF_Output input_op = {TF_GraphOperationByName(model, "input_1"), 0};
    if (input_op.oper == nullptr)
    {
        std::cout << "Can't init input op" << std::endl;
    }

    const std::vector<std::int64_t> input_dims = {1, GAMEMAP_HEIGHT,
                                                  GAMEMAP_WIDTH, CHANNEL_SIZE};
    std::vector<float> input_vals = mcts::encode(state);

    TF_Tensor *input_tensor = tf_utils::CreateTensor(
        TF_FLOAT, input_dims.data(), input_dims.size(), input_vals.data(),
        input_vals.size() * sizeof(float));

    /* prepare output tensor */
    TF_Output out_op = {TF_GraphOperationByName(model, "concatenate_1/concat")};
    if (out_op.oper == nullptr)
    {
        std::cout << "Can't init out_op" << std::endl;
    }

    TF_Tensor *output_tensor = nullptr;

    /* prepare session */
    TF_Status *status = TF_NewStatus();
    TF_SessionOptions *options = TF_NewSessionOptions();
    TF_Session *sess = TF_NewSession(model, options, status);
    TF_DeleteSessionOptions(options);

    if (TF_GetCode(status) != TF_OK)
    {
        TF_DeleteStatus(status);
    }

    /* run session */
    TF_SessionRun(sess,
                  nullptr, // Run options.
                  &input_op, &input_tensor,
                  1, // Input tensors, input tensor values, number of inputs.
                  &out_op, &output_tensor,
                  1,          // Output tensors, output tensor values, number of outputs.
                  nullptr, 0, // Target operations, number of targets.
                  nullptr,    // Run metadata.
                  status      // Output status.
    );

    if (TF_GetCode(status) != TF_OK)
    {
        std::cout << "Error run session";
        TF_DeleteStatus(status);
    }

    TF_CloseSession(sess, status);
    if (TF_GetCode(status) != TF_OK)
    {
        std::cout << "Error close session";
        TF_DeleteStatus(status);
    }

    TF_DeleteSession(sess, status);
    if (TF_GetCode(status) != TF_OK)
    {
        std::cout << "Error delete session";
        TF_DeleteStatus(status);
    }

    const auto probs = static_cast<float *>(TF_TensorData(output_tensor));
    std::vector<float> ret = std::vector<float>(23);
    for (int i = 0; i < 23; i++)
        ret[i] = probs[i];
    return ret;
}
