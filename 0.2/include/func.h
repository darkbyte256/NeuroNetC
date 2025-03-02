#pragma once
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include "../include/neuralnetwork.h"

// sigmoid activation function
double sigmoid(double x);

// Derivative of the sigmoid function
double sigmoid_derivative(double x);

// return a random nor gate struct
nor random_nor_gate();