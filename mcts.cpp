#include "mcts.hpp"
#include "PuyoGame.hpp"
#include "dataProcess.hpp"
#include "define.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>

mcts::MCTS::Node::Node(puyogame::State _state, int _d, float _p, int _r)
    : state(_state), d(_d), p(_p), r(_r), w(0), n(0) {}

float mcts::MCTS::Node::evaluate(const VVI &puyoSeqs, MCTS &parent) {
    if(this->state.isDone()) {
        float value = (this->state.isLose() ? -1.0f : this->r);
        this->w += value;
        this->n++;
        return value;
    }

    if(this->d == EVALUATE_DEPTH) {
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
            this->childNodes.push_back(Node(nextState, this->d + 1, policies[i],
                                            std::max(this->r, reward)));
        }
        return value;
    } else {
        float value =
            this->childNodes[nextChildNode()].evaluate(puyoSeqs, parent);
        this->w += value;
        this->n++;
        return value;
    }
}

int mcts::MCTS::Node::nextChildNode() {
    auto v = nodesToScores(this->childNodes);
    float t = (float)std::accumulate(v.begin(), v.end(), 0);
    std::vector<float> pucbValues;
    for(auto &c : childNodes) {
        if(c.d == 1 && c.n == 0) {
            pucbValues.push_back(100.0);
        } else {
            pucbValues.push_back((c.n != 0 ? (c.w / (float)c.n) : 0.0) +
                                 C_PUCT * c.p *
                                     (sqrtf((float)t) / (float)(1.0 + c.n)));
        }
    }
    int idx =
        std::distance(pucbValues.begin(),
                      std::max_element(pucbValues.begin(), pucbValues.end()));
    return idx;
}

TF_Graph *mcts::MCTS::loadGraph(const char *fileName) {
    mModel = tf_utils::LoadGraph(fileName);
    return mModel;
}

mcts::MCTS::MCTS() : mModel(nullptr), mSess(nullptr) {}

std::vector<int> mcts::MCTS::nodesToScores(const std::vector<Node> &nodes) {
    std::vector<int> ret;
    for(auto &c : nodes)
        ret.push_back(c.n);
    return ret;
}

std::vector<double> mcts::MCTS::randomMcts(puyogame::State state,
                                           float temperature) {
    int childNodeNum = state.legalActions().size();
    std::vector<int> scores(childNodeNum);
    for(int i = 0; i < TRY_COUNT; i++) {
        VVI puyoSeqs = puyogame::State::makePuyoSeqs(TSUMO_SIZE);
        puyogame::State tmp = state;
        auto rootNode = Node(tmp, 0, 0, 0);

        {
            rootNode.evaluate(puyoSeqs, *this);
            for(int j = 0; j < PV_EVALUATE_COUNT; j++) {
                rootNode.evaluate(puyoSeqs, *this);
            }
            for(int j = 0; j < childNodeNum; j++) {
                scores[j] += rootNode.childNodes[j].n;
            }
        }
    }
    if(temperature < 0.001) { // temperature == 0
        int action = std::distance(
            scores.begin(), std::max_element(scores.begin(), scores.end()));
        std::vector<double> ret(scores.size(), 0.0);
        ret[action] = 1.0;
        return ret;
    } else {
        return boltzman(scores, temperature);
    }
}

std::vector<double> mcts::MCTS::normalMcts(puyogame::State state,
                                           const VVI &puyoSeqs,
                                           float temperature) {
    int childNodeNum = state.legalActions().size();
    puyogame::State tmp = state;
    VVI evalPuyoSeqs = puyoSeqs;
    auto rootNode = Node(tmp, 0, 0, 0);
    for(int j = 0; j < PV_EVALUATE_COUNT; j++) {
        rootNode.evaluate(evalPuyoSeqs, *this);
    }
    auto scores = nodesToScores(rootNode.childNodes);
    if(temperature < 0.001) { // temperature == 0
        int action = std::distance(
            scores.begin(), std::max_element(scores.begin(), scores.end()));
        std::vector<double> ret(scores.size(), 0.0);
        ret[action] = 1.0;
        return ret;
    } else {
        return boltzman(scores, temperature);
    }
}

std::vector<double> mcts::MCTS::mcts(puyogame::State state, const VVI &puyoSeqs,
                                     float temperature, int type) {
    if(type == SINGLE) {
        return normalMcts(state, puyoSeqs, temperature);
    } else if(type == RANDOM) {
        return randomMcts(state, temperature);
    } else {
        throw "mcts type error!";
    }
}

std::vector<double> mcts::MCTS::boltzman(const VI &scores,
                                         const float temperature) {
    std::vector<double> ret(scores.size());
    double sum = 0.0;
    for(int i = 0; i < scores.size(); i++) {
        ret[i] = std::pow((double)scores[i], 1.0 / temperature);
        sum += ret[i];
    }
    for(int i = 0; i < scores.size(); i++) {
        ret[i] /= sum;
    }
    return ret;
}

bool mcts::MCTS::prepareSession() {
    /* prepare session */
    mStatus = TF_NewStatus();
    TF_SessionOptions *options = TF_NewSessionOptions();
    mSess = TF_NewSession(mModel, options, mStatus);
    TF_DeleteSessionOptions(options);

    if(TF_GetCode(mStatus) != TF_OK) {
        TF_DeleteStatus(mStatus);
        return false;
    }

    mInputs.resize(1);
    mOutputs.resize(2);

    /* prepare input tensor op */
    TF_Output input_op = {TF_GraphOperationByName(mModel, "input_1"), 0};
    if(input_op.oper == nullptr) {
        return false;
    }
    /* prepare output tensor op */
    TF_Output out_op1 = {TF_GraphOperationByName(mModel, "pi/Softmax")};
    if(out_op1.oper == nullptr) {
        return false;
    }
    TF_Output out_op2 = {TF_GraphOperationByName(mModel, "v/Identity")};
    if(out_op2.oper == nullptr) {
        return false;
    }

    mInputs[0] = input_op;
    mOutputs[0] = out_op1;
    mOutputs[1] = out_op2;

    return true;
}

bool mcts::MCTS::close() {
    if(tf_utils::DeleteSession(mSess, mStatus) != TF_OK) {
        std::cout << "Error delete session";
        TF_DeleteStatus(mStatus);
        return false;
    }

    tf_utils::DeleteGraph(mModel);
    TF_DeleteStatus(mStatus);
    return true;
}

std::pair<std::vector<float>, float>
mcts::MCTS::predict(puyogame::State &state) {
    std::vector<float> input_vals = dataprocess::encode(state);

    TF_Tensor *input_tensor = tf_utils::CreateTensor(
        TF_FLOAT, mInput_dims.data(), mInput_dims.size(), input_vals.data(),
        input_vals.size() * sizeof(float));

    /* prepare session */
    std::vector<TF_Tensor *> input_tensors(1, input_tensor);
    std::vector<TF_Tensor *> output_tensors(2, nullptr);

    /* run session */
    tf_utils::RunSession(mSess, mInputs, input_tensors, mOutputs,
                         output_tensors, mStatus);

    if(TF_GetCode(mStatus) != TF_OK) {
        std::cout << "Error run session";
        TF_DeleteStatus(mStatus);
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

    tf_utils::DeleteTensors(input_tensors);
    tf_utils::DeleteTensors(output_tensors);
    return {ret, val};
}
