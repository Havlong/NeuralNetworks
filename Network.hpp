#ifndef NEURALNETWORKS_NETWORK_HPP
#define NEURALNETWORKS_NETWORK_HPP

#include "Cost.hpp"

#include <chrono>
#include <random>
#include <vector>
#include <algorithm>
#include <iostream>
#include <iomanip>

template<typename T> using layer = vec<T>;

typedef mat<double> weights;
typedef vec<double> activation;
typedef vec<double> biases;

class Network {
private:
    static inline std::mt19937 newRandom() {
        auto duration = std::chrono::steady_clock::now().time_since_epoch();
        auto seed = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

        return std::mt19937(seed);
    }

    std::vector<std::pair<activation, label>> trainingData, testData;
    layer<int> layerSizes;
    layer<biases> layerBiases;
    layer<weights> layerWeights;
    Cost *costFunction = nullptr;

    std::mt19937 rGen = newRandom();
    std::normal_distribution<> distribution = std::normal_distribution<>();

    std::pair<layer<weights>, layer<biases>> backPropagate(const activation &input, label trueLabel);

    int applyMiniBatch(std::vector<std::pair<activation, label>>::const_iterator begin,
                       const std::vector<std::pair<activation, label>>::const_iterator &end,
                       const double &learningRate);

    void resizeLayers();

public:
    activation feedForward(const activation &input);

    void sgd(const int &epochsCount, const int &batchSize, const double &learningRate, bool verbose);

    int evaluate();

    Network(std::vector<std::pair<activation, label>> trainingData, std::vector<std::pair<activation, label>> testData,
            layer<int> layerSizes, Cost *costFunction);
};


#endif
