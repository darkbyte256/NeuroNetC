#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include "../include/func.h"
#include "../include/vec.h"
#include "../include/neuralnetwork.h"

void Ai()
{

    // Initialize the neural network
    neuralnetwork nn;
    init_network(&nn);

    // making training data
    nor *gates = vector_create();
    for (int i = 0; i < 100000; i++)
    {
        vector_add(&gates, random_nor_gate());
    }

    // making test data
    nor *test_gates = vector_create();
    for (int i = 0; i < 10; i++)
    {
        vector_add(&test_gates, random_nor_gate());
    }

    // Train the neural network
    train(&nn, gates);

    // Test the trained neural network
    test(&nn, test_gates);
}

int main()
{
    printf("runing main\n");
    Ai();
    printf("done\n");
}