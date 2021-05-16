#ifndef NEURALNETWORKS_NETWORK_H
#define NEURALNETWORKS_NETWORK_H

#include <vector>
#include <random>
#include <ctime>
#include <algorithm>
#include <iostream>

class Network {
private:
    std::vector<std::pair<std::vector<double>, int>> trainingData, testData;
    std::vector<int> layerSizes;
    std::vector<std::vector<double>> layerBiases;
    std::vector<std::vector<std::vector<double>>> layerWeights;
    std::mt19937 rGen;
    std::uniform_real_distribution<> distribution;

    std::pair<std::vector<std::vector<std::vector<double>>>, std::vector<std::vector<double>>>
    backPropagate(const std::vector<double> &test, int label);

    void applyMiniBatch(const std::vector<std::pair<std::vector<double>, int>> &miniBatch, double learningRate);

    static std::vector<double> costDerivative(const std::vector<double> &output, int label);

    static double sigmoid(double z);

    static std::vector<double> sigmoid(const std::vector<double> &z);

    static double sigmoidPrime(double z);

    static std::vector<double> sigmoidPrime(const std::vector<double> &z);

public:
    std::vector<double> feedForward(std::vector<double> input);

    void sgd(int epochsCount, int batchSize, double learningRate);

    double evaluate();

    Network(std::vector<std::pair<std::vector<double>, int>> trainingData,
            std::vector<std::pair<std::vector<double>, int>> testData,
            std::vector<int> layerSizes);
};


#endif
