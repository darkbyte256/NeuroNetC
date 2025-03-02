#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include "../include/neuralnetwork.h"
#include "../include/func.h"

// typedef struct
// {
//     /* data */
//     double weights[2]; // Weights for two inputs
//     double bias;       // Bias term
// } neuralnetwork;

void init_network(neuralnetwork *nn)
{
    nn->weights[0] = (double)rand() / RAND_MAX; // random weight for input1
    nn->weights[1] = (double)rand() / RAND_MAX; // random weight for input2
    nn->bias = (double)rand() / RAND_MAX;       // random bias
}

// Forward pass function
double forward(neuralnetwork *nn, double input[2])
{
    // Weighted sum of inputs + bias
    double weighted_sum = (input[0] * nn->weights[0]) + (input[1] * nn->weights[1]) + nn->bias;
    // Apply sigmoid activation function
    return sigmoid(weighted_sum);
}

// Backward pass function (update weights and bias using gradient descent)
void backward(neuralnetwork *nn, double input[2], double target, double output)
{
    double error = output - target;
    double d_output = error * sigmoid_derivative(output);

    // update weights and bias
    nn->weights[0] -= LEARNING_RATE * d_output * input[0];
    nn->weights[1] -= LEARNING_RATE * d_output * input[1];
    nn->bias -= LEARNING_RATE * d_output;
}

void train(neuralnetwork *nn, double inputs[4][2], double targets[4])
{
    for (int epoch = 0; epoch < EPOCHS; epoch++)
    {
        double total_error = 0.0;

        // loop though each training example
        for (int i = 0; i < 4; i++)
        {
            double *input = inputs[i];
            double target = targets[i];

            // Forward pass
            double output = forward(nn, input);

            // backward pass
            backward(nn, input, target, output);

            // calculate and accumulate the error
            total_error += (output - target) * (output - target);
        }

        if (epoch % 1000 == 0)
        {
            printf("Epoch %d, Total error: %f\n", epoch, total_error);
        }
    }
}

void test(neuralnetwork *nn, double inputs[4][2], double targets[4])
{
    printf("\nTesting the trained neural network:\n");
    for (int i = 0; i < 4; i++)
    {
        double *input = inputs[i];
        double target = targets[i];

        // predict output for given input
        double prediction = forward(nn, input);

        // print the result
        printf("Input: [%f, %f], Prediction: %f, Target: %f\n", input[0], input[1], prediction, target);
    }
}