#ifndef NEURALNETWORKS_NETWORK_H
#define NEURALNETWORKS_NETWORK_H

#include <chrono>
#include <random>
#include <vector>
#include <algorithm>
#include <iostream>
#include <iomanip>

template<typename T> using vec = std::vector<T>;
template<typename T> using mat = vec<vec<T>>;

template<typename T> using layer = vec<T>;

typedef mat<double> weights;
typedef vec<double> activation;
typedef vec<double> biases;
typedef int label;

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

    std::mt19937 rGen = newRandom();
    std::normal_distribution<> distribution = std::normal_distribution<>();

    std::pair<layer<weights>, layer<biases>> backPropagate(const activation &input, label trueLabel);

    int applyMiniBatch(const std::vector<std::pair<activation, label>> &miniBatch, const double &learningRate);

    void resizeLayers();

    static vec<double> costDerivative(const activation &output, const label &trueLabel);

    static double sigmoid(const double &z);

    static vec<double> sigmoid(const vec<double> &z);

    static double sigmoidPrime(const double &z);

    static vec<double> sigmoidPrime(const vec<double> &z);

public:
    activation feedForward(const activation &input);

    void sgd(const int &epochsCount, const int &batchSize, const double &learningRate);

    int evaluate();

    Network(std::vector<std::pair<activation, label>> trainingData, std::vector<std::pair<activation, label>> testData,
            layer<int> layerSizes);
};


#endif
