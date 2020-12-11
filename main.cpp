#include "tf_utils.hpp"
#include <iostream>
#include <stdio.h>
#include <tensorflow/c/c_api.h>

#define MODEL_FILENAME                                                         \
    "C:/Users/rokahikou/Ohsuga_lab/AlphaPuyo/build/resources/best.pb"

static int displayGraphInfo() {
    printf("%s\n", MODEL_FILENAME);
    TF_Graph *graph = tf_utils::LoadGraph(MODEL_FILENAME);
    if(graph == nullptr) {
        std::cout << "Can't load graph" << std::endl;
        return 1;
    }

    size_t pos = 0;
    TF_Operation *oper;
    printf("--- graph info ---\n");
    while((oper = TF_GraphNextOperation(graph, &pos)) != nullptr) {
        std::cout << TF_OperationName(oper) << std::endl;
    }
    printf("--- graph info ---\n");
    TF_DeleteGraph(graph);
    return 0;
}

std::vector<float> prepare_input() {
    std::vector<float> ret(1 * 13 * 6 * 21, 1.0);
    return ret;
}

int main() {
    printf("Hello from TensorFlow C library version %s\n", TF_Version());
    displayGraphInfo();

    TF_Graph *graph = tf_utils::LoadGraph(MODEL_FILENAME);
    if(graph == nullptr) {
        std::cout << "Can't load graph" << std::endl;
        int a;
        std::cin >> a;
        return 1;
    }

    /* prepare input tensor */
    TF_Output input_op = {TF_GraphOperationByName(graph, "input_1"), 0};
    if(input_op.oper == nullptr) {
        std::cout << "Can't init input op" << std::endl;
        int a;
        std::cin >> a;
        return 2;
    }

    const std::vector<std::int64_t> input_dims = {1, 13, 6, 21};
    std::vector<float> input_vals = prepare_input();

    TF_Tensor *input_tensor = tf_utils::CreateTensor(
        TF_FLOAT, input_dims.data(), input_dims.size(), input_vals.data(),
        input_vals.size() * sizeof(float));

    /* prepare output tensor */
    TF_Output out_op1 = {TF_GraphOperationByName(graph, "pi/Softmax")};
    if(out_op1.oper == nullptr) {
        std::cout << "Can't init out_op1" << std::endl;
        int a;
        std::cin >> a;
        return 3;
    }

    TF_Tensor *output_tensor1 = nullptr;

    TF_Output out_op2 = {TF_GraphOperationByName(graph, "v/Identity")};
    if(out_op2.oper == nullptr) {
        std::cout << "Can't init out_op2" << std::endl;
        int a;
        std::cin >> a;
        return 3;
    }

    TF_Tensor *output_tensor2 = nullptr;

    /* prepare session */
    TF_Status *status = TF_NewStatus();
    TF_SessionOptions *options = TF_NewSessionOptions();
    TF_Session *sess = TF_NewSession(graph, options, status);
    TF_DeleteSessionOptions(options);

    if(TF_GetCode(status) != TF_OK) {
        TF_DeleteStatus(status);
        int a;
        std::cin >> a;
        return 4;
    }

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
    tf_utils::RunSession(sess, inputs, input_tensors, outputs, output_tensors,
                         status);

    if(TF_GetCode(status) != TF_OK) {
        std::cout << "Error run session";
        TF_DeleteStatus(status);
        int a;
        std::cin >> a;
        return 5;
    }

    TF_CloseSession(sess, status);
    if(TF_GetCode(status) != TF_OK) {
        std::cout << "Error close session";
        TF_DeleteStatus(status);
        int a;
        std::cin >> a;
        return 6;
    }

    TF_DeleteSession(sess, status);
    if(TF_GetCode(status) != TF_OK) {
        std::cout << "Error delete session";
        TF_DeleteStatus(status);
        int a;
        std::cin >> a;
        return 7;
    }

    const auto probs = static_cast<float *>(TF_TensorData(output_tensors[0]));

    float sum = 0;
    std::cout << "pi: ";
    for(int i = 0; i < 22; i++) {
        std::cout << probs[i] << " ";
        sum += probs[i];
    }
    std::cout << std::endl;
    std::cout << "sum:" << sum << std::endl;

    const auto value = static_cast<float *>(TF_TensorData(output_tensors[1]));

    std::cout << "v: ";
    std::cout << value[0] << std::endl;
    std::cout << std::endl;

    TF_DeleteTensor(input_tensor);
    TF_DeleteTensor(output_tensor1);
    TF_DeleteTensor(output_tensor2);
    TF_DeleteGraph(graph);
    TF_DeleteStatus(status);

    int a;
    std::cin >> a;
    return 0;
}