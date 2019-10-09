#ifndef NEURALNETWORKS_NETWORK_H
#define NEURALNETWORKS_NETWORK_H

#include <vector>
#include <random>
#include <ctime>

class Network {
private:
    std::vector<std::pair<std::vector<double>, int>> trainingData, testData;
    std::vector<int> layerSizes;
    std::vector<std::vector<double>> layerBiases;
    std::vector<std::vector<std::vector<double>>> layerWeights;
    std::mt19937 rGen;
    std::uniform_real_distribution<> arthas;

    std::pair<std::vector<std::vector<double>>, std::vector<double>>
    backPropagate(const std::vector<double> &test, int label);

    void applyMiniBatch(const std::vector<std::pair<std::vector<double>, int>> &miniBatch, double learningRate);

    std::vector<double> costDerivative(const std::vector<double> &output, int label);

    double sigmoid(double z);

    std::vector<double> sigmoid(const std::vector<double> &z);

    double sigmoidPrime(double z);

    static std::vector<double> dot(const std::vector<std::vector<double>> &m, const std::vector<double> &v);

    static std::vector<double> sum(const std::vector<double> &l, const std::vector<double> &r);

    std::vector<double> sigmoidPrime(const std::vector<double> &z);

public:
    std::vector<double> feedForward(std::vector<double> input);

    void sgd(int epochsCount, int batchSize, double learningRate);

    int evaluate();

    Network(std::vector<std::pair<std::vector<double>, int>> trainingData,
            std::vector<std::pair<std::vector<double>, int>> testData,
            std::vector<int> layerSizes);
};


#endif
