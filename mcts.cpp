#include "mcts.hpp"
#include "PuyoGame.hpp"
#include "define.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

mcts::Node::Node(puyogame::State state, int d, float p, int r, int turn,
                 TF_Graph *model, TF_Session *sess) {
    this->state = state;
    this->p = p;
    this->d = d;
    this->r = r;
    this->w = 0;
    this->n = 0;
    this->turn = turn;
    this->model = model;
    this->sess = sess;
}

float mcts::Node::evaluate(const VVI &puyoSeqs) {
    if(this->state.isDone() || this->turn == MAX_STEP) {
        float value = (this->state.isLose() ? -1 : r);
        this->w += value;
        this->n++;
        return value;
    }

    if(d == EVALUATE_DEPTH) {
        float value = this->state.calcMaxReward();
        this->w += value;
        this->n++;
        return value;
    }

    if(this->childNodes.empty()) {
        auto [policies, value] = MCTS::predict(this->state, model, sess);
        // std::cout << "predict value: " << value << std::endl;
        // std::cout << "predict policies: ";
        // for (auto p : policies)
        //     std::cout << p << " ";
        // std::cout << std::endl;
        this->w += value;
        this->n++;
        VI actions = this->state.legalActions();
        for(int i = 0; i < actions.size(); i++) {
            int reward = 0;
            puyogame::State nextState =
                this->state.next(actions[i], puyoSeqs[this->d], reward);
            this->childNodes.push_back(std::unique_ptr<Node>(new Node(
                nextState, this->d + 1, policies[actions[i]],
                std::max(this->r, reward), this->turn + 1, model, sess)));
        }
        return value;
    } else {
        float value = this->childNodes[nextChildNode()]->evaluate(puyoSeqs);
        this->w += value;
        this->n++;
        return value;
    }
}

int mcts::Node::nextChildNode() {
    auto v = MCTS::nodesToScores(childNodes);
    int t = std::accumulate(v.begin(), v.end(), 0);
    std::vector<float> pucbValues;
    for(auto &c : childNodes) {
        pucbValues.push_back((c->n ? (c->w / (float)c->n) : 0.0) +
                             (w / (float)n) * c->p * sqrt(t) /
                                 (float)(1 + c->n));
    }
    // std::cout << "w : " << w << std::endl;
    // std::cout << "=== pucbValues ===" << std::endl;
    // for (int i = 0; i < pucbValues.size(); i++)
    // {
    //     std::cout << pucbValues[i] << " ";
    // }
    // std::cout << std::endl;

    int idx =
        std::distance(pucbValues.begin(),
                      std::max_element(pucbValues.begin(), pucbValues.end()));
    return idx;
}

TF_Graph *mcts::MCTS::loadGraph(const char *fileName) {
    model = tf_utils::LoadGraph(fileName);
    return model;
}

std::vector<int> mcts::MCTS::nodesToScores(
    const std::vector<std::unique_ptr<mcts::Node>> &nodes) {
    std::vector<int> ret;
    for(auto &c : nodes)
        ret.push_back(c->n);
    return ret;
}

std::vector<int> mcts::MCTS::mctsScores(puyogame::State state,
                                        float temperature) {
    int childNodeNum = state.legalActions().size();
    std::vector<int> childNodesPlayCounts(childNodeNum);
    for(int i = 0; i < TRY_COUNT; i++) {
        // std::cout << "TRY " << i << std::endl;
        VVI puyoSeqs = puyogame::State::makePuyoSeqs(10);
        puyogame::State tmp = state;
        Node rootNode = Node(tmp, 0, 0, 0, tmp.turn, model, sess);
        for(int j = 0; j < PV_EVALUATE_COUNT; j++) {
            rootNode.evaluate(puyoSeqs);
            // std::cout << "evaluate " << j << std::endl;
        }
        for(int j = 0; j < childNodeNum; j++) {
            childNodesPlayCounts[j] += rootNode.childNodes[j]->n;
        }
    }
    return childNodesPlayCounts;
}

std::vector<float> mcts::MCTS::encode(const puyogame::State &state) {
    return data2Binary(state.gameMap, state.puyos[0], state.puyos[1],
                       state.turn);
}

std::vector<float> mcts::MCTS::data2Binary(const VVI &stage, VI nowPuyo,
                                           VI nextPuyo, int turn) {
    std::vector<std::vector<std::vector<int>>> mat(
        GAMEMAP_HEIGHT, std::vector<std::vector<int>>(
                            GAMEMAP_WIDTH, std::vector<int>(CHANNEL_SIZE)));
    for(int i = 0; i < GAMEMAP_HEIGHT; i++)
        for(int j = 0; j < GAMEMAP_WIDTH; j++)
            mat[i][j][stage[i][j]] = 1;

    VI puyoData = VI();
    for(int i = 0; i < 2; i++)
        puyoData.push_back(nowPuyo[i]);
    for(int i = 0; i < 2; i++)
        puyoData.push_back(nextPuyo[i]);
    for(int k = 0; k < 4; k++)
        for(int i = 0; i < GAMEMAP_HEIGHT; i++)
            for(int j = 0; j < GAMEMAP_WIDTH; j++)
                mat[i][j][puyoData[k] + PUYO_COLOR + k * PUYO_COLOR];
    std::vector<float> ret(GAMEMAP_HEIGHT * GAMEMAP_WIDTH * CHANNEL_SIZE);
    int cur = 0;
    for(int i = 0; i < GAMEMAP_HEIGHT; i++)
        for(int j = 0; j < GAMEMAP_WIDTH; j++)
            for(int k = 0; k < CHANNEL_SIZE; k++)
                ret[cur++] = mat[i][j][k];
    return ret;
}

bool mcts::MCTS::prepareSession() {
    /* prepare session */
    TF_Status *status = TF_NewStatus();
    TF_SessionOptions *options = TF_NewSessionOptions();
    sess = TF_NewSession(model, options, status);
    TF_DeleteSessionOptions(options);

    if(TF_GetCode(status) != TF_OK) {
        TF_DeleteStatus(status);
        return false;
    }
    return true;
}

bool mcts::MCTS::closeSession() {
    TF_Status *status = TF_NewStatus();
    TF_CloseSession(sess, status);
    if(TF_GetCode(status) != TF_OK) {
        std::cout << "Error close session";
        TF_DeleteStatus(status);
        return false;
    }

    TF_DeleteSession(sess, status);
    if(TF_GetCode(status) != TF_OK) {
        std::cout << "Error delete session";
        TF_DeleteStatus(status);
        return false;
    }
    return true;
}

std::pair<std::vector<float>, float>
mcts::MCTS::predict(const puyogame::State &state, TF_Graph *model,
                    TF_Session *sess) {
    /* prepare input tensor */
    TF_Output input_op = {TF_GraphOperationByName(model, "input_1"), 0};
    if(input_op.oper == nullptr) {
        std::cout << "Can't init input op" << std::endl;
    }

    const std::vector<std::int64_t> input_dims = {1, GAMEMAP_HEIGHT,
                                                  GAMEMAP_WIDTH, CHANNEL_SIZE};
    std::vector<float> input_vals = mcts::MCTS::encode(state);

    TF_Tensor *input_tensor = tf_utils::CreateTensor(
        TF_FLOAT, input_dims.data(), input_dims.size(), input_vals.data(),
        input_vals.size() * sizeof(float));

    /* prepare output tensor */
    TF_Output out_op1 = {TF_GraphOperationByName(model, "pi/Softmax")};
    if(out_op1.oper == nullptr) {
        std::cout << "Can't init out_op1" << std::endl;
    }

    TF_Tensor *output_tensor1 = nullptr;

    TF_Output out_op2 = {TF_GraphOperationByName(model, "v/Identity")};
    if(out_op2.oper == nullptr) {
        std::cout << "Can't init out_op2" << std::endl;
    }

    TF_Tensor *output_tensor2 = nullptr;

    /* prepare session */
    std::vector<TF_Output> inputs;
    std::vector<TF_Tensor *> input_tensors;
    std::vector<TF_Output> outputs;
    std::vector<TF_Tensor *> output_tensors;

    inputs.push_back(input_op);
    input_tensors.push_back(input_tensor);
    outputs.push_back(out_op1);
    output_tensors.push_back(output_tensor1);
    outputs.push_back(out_op2);
    output_tensors.push_back(output_tensor2);

    /* run session */
    TF_Status *status = TF_NewStatus();
    tf_utils::RunSession(sess, inputs, input_tensors, outputs, output_tensors,
                         status);

    if(TF_GetCode(status) != TF_OK) {
        std::cout << "Error run session";
        TF_DeleteStatus(status);
    }

    const auto probs = static_cast<float *>(TF_TensorData(output_tensors[0]));
    std::vector<float> ret = std::vector<float>(ACTION_KIND);
    for(int i = 0; i < ACTION_KIND; i++)
        ret[i] = probs[i];
    const auto value = static_cast<float *>(TF_TensorData(output_tensors[1]));
    auto val = value[0];

    TF_DeleteTensor(input_tensor);
    TF_DeleteTensor(output_tensor1);
    TF_DeleteTensor(output_tensor2);
    TF_DeleteStatus(status);

    return {ret, val};
}
