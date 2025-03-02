#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include "../include/func.h"
#include "../include/vec.h"
#include "../include/neuralnetwork.h"

void test1()
{
    printf("test1\n");
    int num1 = 0;
    int num2 = 0;
    printf("num1: ");
    scanf("%d", &num1);

    printf("num2: ");
    scanf("%d", &num2);

    printf("%d + %d = %d\n", num1, num2, add(num1, num2));
}

void test2()
{
    int *vec_int = vector_create();
    int max_vec_num = 1000000000;
    srand(time(NULL));

    for (int i = 0; i < max_vec_num; i++)
    {
        vector_add(&vec_int, rand());
    }

    for (int i = 0; i < max_vec_num; i++)
    {
        // printf(" %d ", vec_int[i]);
    }

    scanf("%c");

    vector_free(vec_int);
}

void Ai()
{
    // Input data for the NOR gate (4 examples, 2 inputs)
    double inputs[4][2] = {
        {0.0, 0.0}, // 0 NOR 0 = 1
        {0.0, 1.0}, // 0 NOR 1 = 0
        {1.0, 0.0}, // 1 NOR 0 = 0
        {1.0, 1.0}  // 1 NOR 1 = 0
    };

    // Target outputs for NOR gate
    double targets[4] = {1.0, 0.0, 0.0, 0.0};

    // Initialize the neural network
    neuralnetwork nn;
    init_network(&nn);

    // Train the neural network
    train(&nn, inputs, targets);

    // Test the trained neural network
    test(&nn, inputs, targets);
}

int main()
{
    printf("runing main\n");
    // test1();
    // test2();
    Ai();
    printf("done\n");
}