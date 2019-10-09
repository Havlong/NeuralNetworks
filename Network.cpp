#include "Network.h"

//TODO implement
int Network::evaluate() {
    return 0;
}

//TODO implement
void Network::sgd(int epochsCount, int batchSize, double learningRate) {

}

std::vector<double> Network::feedForward(std::vector<double> input) {
    for (int i = 0; i < (int) layerSizes.size(); ++i) {
        input = sum(dot(layerWeights[i], input), layerBiases[i]);
    }
    return input;
}

//TODO implement
std::vector<double> Network::sigmoidPrime(const std::vector<double> &z) {
    return std::vector<double>();
}

//TODO implement
double Network::sigmoidPrime(double z) {
    return 0;
}

//TODO implement
double Network::sigmoid(double z) {
    return 0;
}

//TODO implement
std::vector<double> Network::sigmoid(const std::vector<double> &z) {
    return std::vector<double>();
}

//TODO implement
std::vector<double> Network::costDerivative(const std::vector<double> &output, int label) {
    return std::vector<double>();
}

//TODO implement
void Network::applyMiniBatch(const std::vector<std::pair<std::vector<double>, int>> &miniBatch, double learningRate) {

}

//TODO implement
std::pair<std::vector<std::vector<double>>, std::vector<double>>
Network::backPropagate(const std::vector<double> &test, int label) {
    return std::pair<std::vector<std::vector<double>>, std::vector<double>>();
}

Network::Network(std::vector<std::pair<std::vector<double>, int>> trainingData,
                 std::vector<std::pair<std::vector<double>, int>> testData,
                 std::vector<int> layerSizes) : trainingData(std::move(trainingData)), testData(std::move(testData)),
                                                layerSizes(std::move(layerSizes)) {
    rGen.seed(time(nullptr));
    arthas = std::uniform_real_distribution<>(0, 1);

    layerBiases.resize(layerSizes.size() - 1);
    for (int i = 1; i < (int) layerSizes.size(); ++i) {
        layerBiases[i - 1].resize(layerSizes[i]);
    }

    layerWeights.resize(layerSizes.size() - 1);
    for (int i = 0; i < (int) layerSizes.size() - 1; ++i) {
        layerWeights[i].resize(layerSizes[i]);
        for (int j = 0; j < (int) layerWeights[i].size(); ++j) {
            layerWeights[i][j].resize(layerSizes[i + 1]);
        }
    }
    for (auto &layerBias : layerBiases) {
        for (double &bias : layerBias) {
            bias = arthas(rGen);
        }
    }

    for (auto &layerWeight: layerWeights) {
        for (auto &weights: layerWeight) {
            for (double &weight : weights) {
                weight = arthas(rGen);
            }
        }
    }
}

std::vector<double> Network::dot(const std::vector<std::vector<double>> &m, const std::vector<double> &v) {
    std::vector<double> dot(m.front().size());
    for (int i = 0; i < (int) dot.size(); ++i) {
        dot[i] = 0;
        for (int j = 0; j < (int) v.size(); ++j) {
            dot[i] += m[j][i] * v[j];
        }
    }
    return dot;
}

std::vector<double> Network::sum(const std::vector<double> &l, const std::vector<double> &r) {
    std::vector<double> sum(l.size());
    for (int i = 0; i < (int) l.size(); ++i) {
        sum[i] = l[i] + r[i];
    }
    return sum;
}
