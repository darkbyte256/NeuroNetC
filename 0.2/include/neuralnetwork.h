#pragma once
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include "../include/func.h"
#include "../include/vec.h"

#define LEARNING_RATE 0.0001
#define EPOCHS 100000000

// main struct for the neuralnetwork
typedef struct
{
    /* data */
    double weights[2]; // Weights for two inputs
    double bias;       // Bias term
} neuralnetwork;

// nor gate
typedef struct
{
    /* data */
    double a;
    double b;
    double out;
} nor;
// initulize neuralnetwork
void init_network(neuralnetwork *nn);
// neuralnetwork forward pass function
double forward(neuralnetwork *nn, nor gate);
// neuralnetwork backward pass function (update weights and bias using gradient descent)
void backward(neuralnetwork *nn, nor gate, double output);
// neuralnetwork training function
void train(neuralnetwork *nn, nor *gates);
// neuralnetwork test function
void test(neuralnetwork *nn, nor *gates);