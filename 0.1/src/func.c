#include <math.h>
#include "../include/func.h"

int add(int num1, int num2)
{
    return num1 + num2;
}

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
