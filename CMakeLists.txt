cmake_minimum_required(VERSION 3.15)
project(NeuralNetworks)

set(CMAKE_CXX_STANDARD 17)

add_library(HavlongNN Network.cpp MNISTReader.cpp MathUtils.cpp Cost.cpp)

add_executable(NeuralNetworks main.cpp)
target_link_libraries(NeuralNetworks HavlongNN)
