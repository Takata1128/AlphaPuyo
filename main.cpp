#include "tf_utils.hpp"
#include <iostream>
#include <stdio.h>
#include <tensorflow/c/c_api.h>

#define MODEL_FILENAME \
    "C:/Users/s.takata/Documents/tensorflow_cpp/build/resources/best.pb"

static int displayGraphInfo()
{
    printf("%s\n", MODEL_FILENAME);
    TF_Graph *graph = tf_utils::LoadGraph(MODEL_FILENAME);
    if (graph == nullptr)
    {
        std::cout << "Can't load graph" << std::endl;
        return 1;
    }

    size_t pos = 0;
    TF_Operation *oper;
    printf("--- graph info ---\n");
    while ((oper = TF_GraphNextOperation(graph, &pos)) != nullptr)
    {
        std::cout << TF_OperationName(oper) << std::endl;
    }
    printf("--- graph info ---\n");
    TF_DeleteGraph(graph);
    return 0;
}

std::vector<float> prepare_input()
{
    std::vector<float> ret(1 * 13 * 6 * 21, 1.0);
    return ret;
}

int main()
{
    printf("Hello from TensorFlow C library version %s\n", TF_Version());
    displayGraphInfo();

    TF_Graph *graph = tf_utils::LoadGraph(MODEL_FILENAME);
    if (graph == nullptr)
    {
        std::cout << "Can't load graph" << std::endl;
        int a;
        std::cin >> a;
        return 1;
    }

    /* prepare input tensor */
    TF_Output input_op = {TF_GraphOperationByName(graph, "input_1"), 0};
    if (input_op.oper == nullptr)
    {
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
    TF_Output out_op = {TF_GraphOperationByName(graph, "concatenate_1/concat")};
    if (out_op.oper == nullptr)
    {
        std::cout << "Can't init out_op" << std::endl;
        int a;
        std::cin >> a;
        return 3;
    }

    TF_Tensor *output_tensor = nullptr;

    /* prepare session */
    TF_Status *status = TF_NewStatus();
    TF_SessionOptions *options = TF_NewSessionOptions();
    TF_Session *sess = TF_NewSession(graph, options, status);
    TF_DeleteSessionOptions(options);

    if (TF_GetCode(status) != TF_OK)
    {
        TF_DeleteStatus(status);
        int a;
        std::cin >> a;
        return 4;
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
        int a;
        std::cin >> a;
        return 5;
    }

    TF_CloseSession(sess, status);
    if (TF_GetCode(status) != TF_OK)
    {
        std::cout << "Error close session";
        TF_DeleteStatus(status);
        int a;
        std::cin >> a;
        return 6;
    }

    TF_DeleteSession(sess, status);
    if (TF_GetCode(status) != TF_OK)
    {
        std::cout << "Error delete session";
        TF_DeleteStatus(status);
        int a;
        std::cin >> a;
        return 7;
    }

    const auto probs = static_cast<float *>(TF_TensorData(output_tensor));

    float sum = 0;
    std::cout << "pi: ";
    for (int i = 0; i < 22; i++)
    {
        std::cout << probs[i] << " ";
        sum += probs[i];
    }
    std::cout << std::endl;
    std::cout << "sum:" << sum << std::endl;

    std::cout << "v: ";
    std::cout << probs[22] << std::endl;
    std::cout << std::endl;

    TF_DeleteTensor(input_tensor);
    TF_DeleteTensor(output_tensor);
    TF_DeleteGraph(graph);
    TF_DeleteStatus(status);

    int a;
    std::cin >> a;
    return 0;
}