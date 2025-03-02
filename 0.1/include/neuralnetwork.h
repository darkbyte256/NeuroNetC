#pragma once

#define LEARNING_RATE 0.1
#define EPOCHS 10000

typedef struct
{
    /* data */
    double weights[2]; // Weights for two inputs
    double bias;       // Bias term
} neuralnetwork;

void init_network(neuralnetwork *nn);

double forward(neuralnetwork *nn, double input[2]);

void backward(neuralnetwork *nn, double input[2], double target, double output);

void train(neuralnetwork *nn, double inputs[4][2], double targets[4]);

void test(neuralnetwork *nn, double inputs[4][2], double targets[4]);