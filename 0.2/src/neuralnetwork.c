#include "../include/neuralnetwork.h"

// initulize neuralnetwork
void init_network(neuralnetwork *nn)
{
    nn->weights[0] = (double)rand() / RAND_MAX; // random weight for input1
    nn->weights[1] = (double)rand() / RAND_MAX; // random weight for input2
    nn->bias = (double)rand() / RAND_MAX;       // random bias
}

// Forward pass function
double forward(neuralnetwork *nn, nor gate)
{
    // Weighted sum of inputs + bias
    double weighted_sum = (gate.a * nn->weights[0]) + (gate.b * nn->weights[1]) + nn->bias;
    // Apply sigmoid activation function
    return sigmoid(weighted_sum);
}

// Backward pass function (update weights and bias using gradient descent)
void backward(neuralnetwork *nn, nor gate, double output)
{
    double error = output - gate.out;
    double d_output = error * sigmoid_derivative(output);

    // update weights and bias
    nn->weights[0] -= LEARNING_RATE * d_output * gate.a;
    nn->weights[1] -= LEARNING_RATE * d_output * gate.b;
    nn->bias -= LEARNING_RATE * d_output;
}

// train neuralnetwork
void train(neuralnetwork *nn, nor *gates)
{
    for (int epoch = 0; epoch < EPOCHS; epoch++)
    {
        double total_error = 0.0;

        // loop though each training example
        for (int i = 0; i < 4; i++)
        {
            nor gate = gates[i];

            // Forward pass
            double output = forward(nn, gate);

            // backward pass
            backward(nn, gate, output);

            // calculate and accumulate the error
            total_error += (output - gate.out) * (output - gate.out);
        }

        if (epoch % 10000 == 0)
        {
            printf("Epoch %d, Total error: %f\n", epoch, total_error);
        }
    }
}

// test neuralnetwork
void test(neuralnetwork *nn, nor *gates)
{
    printf("\nTesting the trained neural network:\n");
    for (int i = 0; i < vector_size(gates); i++)
    {
        nor input = gates[i];

        // predict output for given input
        double prediction = forward(nn, input);

        // print the result
        printf("Input: [%f, %f], Prediction: %f, Target: %f\n", input.a, input.b, prediction, input.out);
    }
}