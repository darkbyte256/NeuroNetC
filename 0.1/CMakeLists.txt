cmake_minimum_required(VERSION 3.10.0)
project(app VERSION 0.1.0 LANGUAGES C)

add_executable(app src/main.c src/func.c src/vec.c src/neuralnetwork.c)

# Link the math library
target_link_libraries(app m)

