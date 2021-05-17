cmake_minimum_required(VERSION 3.15)
project(NeuralNetworks)

set(CMAKE_CXX_STANDARD 17)

add_executable(NeuralNetworks main.cpp Network.cpp Network.h MNISTReader.cpp MNISTReader.h)