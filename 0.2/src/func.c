#include "../include/func.h"

// sigmoid activation function
double sigmoid(double x)
{
    return 1 / (1 + exp(-x));
}

// Derivative of the sigmoid function
double sigmoid_derivative(double x)
{
    return x * (1 - x);
}

// return a random nor gate struct
nor random_nor_gate()
{
    nor gate;
    gate.a = round((double)rand() / (double)RAND_MAX);
    gate.b = round((double)rand() / (double)RAND_MAX);
    gate.out = !(gate.a + gate.b);

    return gate;
}