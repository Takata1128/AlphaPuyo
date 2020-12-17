#include "mcts.hpp"
#include "PuyoGame.hpp"
#include "define.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>

mcts::MCTS::Node::Node(puyogame::State state, int d, float p, int r, int turn) {
    this->state = state;
    this->p = p;
    this->d = d;
    this->r = r;
    this->w = 0;
    this->n = 0;
    this->turn = turn;
}

float mcts::MCTS::Node::evaluate(const VVI &puyoSeqs, MCTS &parent) {
    if(this->state.isDone() || this->turn == MAX_STEP) {
        float value = (this->state.isLose() ? -1 : this->r);
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
        auto [policies, value] = parent.predict(this->state);
        this->w += value;
        this->n++;
        VI actions = this->state.legalActions();
        for(int i = 0; i < actions.size(); i++) {
            int reward = 0;
            puyogame::State nextState =
                this->state.next(actions[i], puyoSeqs[this->d], reward);
            this->childNodes.push_back(std::unique_ptr<Node>(
                new Node(nextState, this->d + 1, policies[i],
                         std::max(this->r, reward), this->turn + 1)));
        }
        return value;
    } else {
        float value =
            this->childNodes[nextChildNode()]->evaluate(puyoSeqs, parent);
        this->w += value;
        this->n++;
        return value;
    }
}

int mcts::MCTS::Node::nextChildNode() {
    auto v = nodesToScores(childNodes);
    int t = std::accumulate(v.begin(), v.end(), 0);
    std::vector<float> pucbValues;
    for(auto &c : childNodes) {
        pucbValues.push_back((c->n ? (c->w / (float)c->n) : 0.0) +
                             C_PUCT * c->p * sqrt(t) / (float)(1 + c->n));
    }
    int idx =
        std::distance(pucbValues.begin(),
                      std::max_element(pucbValues.begin(), pucbValues.end()));
    return idx;
}

TF_Graph *mcts::MCTS::loadGraph(const char *fileName) {
    model = tf_utils::LoadGraph(fileName);
    return model;
}

mcts::MCTS::MCTS() {
    model = nullptr;
    sess = nullptr;
}

std::vector<int>
mcts::MCTS::nodesToScores(const std::vector<std::unique_ptr<Node>> &nodes) {
    std::vector<int> ret;
    for(auto &c : nodes)
        ret.push_back(c->n);
    return ret;
}

std::vector<int> mcts::MCTS::randomMcts(puyogame::State state,
                                        float temperature) {
    int childNodeNum = state.legalActions().size();
    std::vector<int> childNodesPlayCounts(childNodeNum);
    for(int i = 0; i < TRY_COUNT; i++) {
        VVI puyoSeqs = puyogame::State::makePuyoSeqs(TSUMO_SIZE);
        puyogame::State tmp = state;
        auto rootNode = std::unique_ptr<Node>(new Node(tmp, 0, 0, 0, tmp.turn));
        for(int j = 0; j < PV_EVALUATE_COUNT; j++) {
            rootNode->evaluate(puyoSeqs, *this);
        }
        for(int j = 0; j < childNodeNum; j++) {
            childNodesPlayCounts[j] += rootNode->childNodes[j]->n;
        }
    }
    return childNodesPlayCounts;
}

std::vector<int> mcts::MCTS::normalMcts(puyogame::State state,
                                        const VVI &puyoSeqs,
                                        float temperature) {
    int childNodeNum = state.legalActions().size();
    std::vector<int> childNodesPlayCounts(childNodeNum);
    puyogame::State tmp = state;
    VVI evalPuyoSeqs(TSUMO_SIZE, VI(2));
    for(int i = 0; i < TSUMO_SIZE; i++) {
        for(int j = 0; j < 2; j++) {
            evalPuyoSeqs[i][j] = puyoSeqs[state.turn + i][j];
        }
    }

    auto rootNode = std::unique_ptr<Node>(new Node(tmp, 0, 0, 0, tmp.turn));

    for(int j = 0; j < PV_EVALUATE_COUNT; j++) {
        rootNode->evaluate(evalPuyoSeqs, *this);
    }
    for(int j = 0; j < childNodeNum; j++) {
        childNodesPlayCounts[j] += rootNode->childNodes[j]->n;
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

    VI puyoData = VI(4);
    for(int i = 0; i < 2; i++)
        puyoData[i] = nowPuyo[i];
    for(int i = 0; i < 2; i++)
        puyoData[i + 2] = nextPuyo[i];
    for(int k = 0; k < 4; k++)
        for(int i = 0; i < GAMEMAP_HEIGHT; i++)
            for(int j = 0; j < GAMEMAP_WIDTH; j++)
                mat[i][j][puyoData[k] + PUYO_COLOR + k * PUYO_COLOR] = 1;
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
    TF_DeleteStatus(status);

    inputs.resize(1);
    outputs.resize(2);

    /* prepare input tensor op */
    TF_Output input_op = {TF_GraphOperationByName(model, "input_1"), 0};
    if(input_op.oper == nullptr) {
        return false;
    }
    /* prepare output tensor op */
    TF_Output out_op1 = {TF_GraphOperationByName(model, "pi/Softmax")};
    if(out_op1.oper == nullptr) {
        return false;
    }
    TF_Output out_op2 = {TF_GraphOperationByName(model, "v/Identity")};
    if(out_op2.oper == nullptr) {
        return false;
    }

    inputs[0] = input_op;
    outputs[0] = out_op1;
    outputs[1] = out_op2;

    return true;
}

bool mcts::MCTS::close() {
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

    TF_DeleteGraph(model);
    TF_DeleteStatus(status);
    return true;
}

std::pair<std::vector<float>, float>
mcts::MCTS::predict(puyogame::State &state) {
    std::vector<float> input_vals = mcts::MCTS::encode(state);

    TF_Tensor *input_tensor = tf_utils::CreateTensor(
        TF_FLOAT, input_dims.data(), input_dims.size(), input_vals.data(),
        input_vals.size() * sizeof(float));

    TF_Tensor *output_tensor1 = nullptr;
    TF_Tensor *output_tensor2 = nullptr;

    /* prepare session */
    std::vector<TF_Tensor *> input_tensors(1);
    std::vector<TF_Tensor *> output_tensors(2);

    input_tensors[0] = input_tensor;
    output_tensors[0] = output_tensor1;
    output_tensors[1] = output_tensor2;

    /* run session */
    TF_Status *status = TF_NewStatus();
    tf_utils::RunSession(sess, inputs, input_tensors, outputs, output_tensors,
                         status);

    if(TF_GetCode(status) != TF_OK) {
        std::cout << "Error run session";
        TF_DeleteStatus(status);
    }

    const auto probs = static_cast<float *>(TF_TensorData(output_tensors[0]));

    // 合法手のみ抽出
    float sum = 0;
    auto legalActions = state.legalActions();
    std::vector<float> ret = std::vector<float>(legalActions.size());
    for(int i = 0; i < legalActions.size(); i++) {
        ret[i] = probs[legalActions[i]];
        sum += probs[legalActions[i]];
    }

    // 和が１になるように正規化
    for(int i = 0; i < ret.size(); i++) {
        ret[i] /= sum;
    }

    const auto value = static_cast<float *>(TF_TensorData(output_tensors[1]));
    auto val = value[0];

    TF_DeleteTensor(input_tensor);
    TF_DeleteTensor(output_tensor1);
    TF_DeleteTensor(output_tensor2);
    TF_DeleteStatus(status);
    return {ret, val};
}
