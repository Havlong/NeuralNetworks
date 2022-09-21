#ifndef NEURALNETWORKS_NETWORK_H
#define NEURALNETWORKS_NETWORK_H

#include <chrono>
#include <random>
#include <vector>
#include <algorithm>
#include <iostream>

class Network {
private:
    static inline std::mt19937 newRandom() {
        auto duration = std::chrono::steady_clock::now().time_since_epoch();
        auto seed = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

        return std::mt19937(seed);
    }

    std::vector<std::pair<std::vector<double>, int>> trainingData, testData;
    std::vector<int> layerSizes;
    std::vector<std::vector<double>> layerBiases;
    std::vector<std::vector<std::vector<double>>> layerWeights;
    std::mt19937 rGen = newRandom();
    std::normal_distribution<> distribution = std::normal_distribution<>();

    std::pair<std::vector<std::vector<std::vector<double>>>, std::vector<std::vector<double>>>
    backPropagate(const std::vector<double> &test, int label);

    int applyMiniBatch(const std::vector<std::pair<std::vector<double>, int>> &miniBatch, const double &learningRate);

    void resizeLayers();

    static std::vector<double> costDerivative(const std::vector<double> &output, const int &label);

    static double sigmoid(const double &z);

    static std::vector<double> sigmoid(const std::vector<double> &z);

    static double sigmoidPrime(const double &z);

    static std::vector<double> sigmoidPrime(const std::vector<double> &z);

public:
    std::vector<double> feedForward(const std::vector<double> &input);

    void sgd(const int &epochsCount, const int &batchSize, const double &learningRate);

    double evaluate();

    Network(std::vector<std::pair<std::vector<double>, int>> trainingData,
            std::vector<std::pair<std::vector<double>, int>> testData, std::vector<int> layerSizes);
};


#endif
