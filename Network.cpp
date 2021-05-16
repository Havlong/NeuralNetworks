#include "Network.h"

/**
 * Feed-forwards test data into the neural net and determines the amount of correct answers
 *
 * @return Amount of correct answers
 */
int Network::evaluate() {
    int amount = 0;
    for (auto &[x, y]: testData) {
        auto result = feedForward(x);
        int resultLabel = std::max_element(result.begin(), result.end()) - result.begin();
        amount += (resultLabel == y);
    }
    return amount;
}

//TODO implement SGD mechanisms
void Network::sgd(int epochsCount, int batchSize, double learningRate) {

}

std::vector<double> Network::feedForward(std::vector<double> input) {
    for (int i = 0; i < (int) layerSizes.size(); ++i) {
        input = sum(dot(layerWeights[i], input), layerBiases[i]);
    }
    return input;
}

double Network::sigmoid(double z) {
    return 1.0 / (1.0 - exp(-z));
}

std::vector<double> Network::sigmoid(const std::vector<double> &z) {
    std::vector<double> result(z.size());
    std::transform(z.begin(), z.end(), result.begin(), std::ptr_fun<double, double>(Network::sigmoid));
    return result;
}

double Network::sigmoidPrime(double z) {
    return sigmoid(z) * (1.0 - sigmoid(z));
}

std::vector<double> Network::sigmoidPrime(const std::vector<double> &z) {
    std::vector<double> result(z.size());
    std::transform(z.begin(), z.end(), result.begin(), std::ptr_fun<double, double>(Network::sigmoidPrime));
    return result;
}

std::vector<double> Network::costDerivative(const std::vector<double> &output, int label) {
    std::vector<double> result(output.size());
    for (int i = 0; i < output.size(); ++i) {
        result[i] = output[i] - label == i ? 1.0 : 0.0;
    }
    return result;
}

void Network::applyMiniBatch(const std::vector<std::pair<std::vector<double>, int>> &miniBatch, double learningRate) {
    // Initialize biases shape
    std::vector<std::vector<double>> deltaBiases(layerBiases.size());
    for (int i = 0; i < deltaBiases.size(); ++i) {
        deltaBiases[i] = std::vector<double>(layerBiases[i].size());
    }

    // initialize weights shape
    std::vector<std::vector<std::vector<double>>> deltaWeights(layerWeights.size());
    for (int i = 0; i < deltaWeights.size(); ++i) {
        deltaWeights[i] = std::vector<std::vector<double>>(layerWeights[i].size());
        for (int j = 0; j < deltaWeights[i].size(); ++j) {
            deltaWeights[i][j] = std::vector<double>(layerWeights[i][j].size());
        }
    }

    for (const auto &[x, y]: miniBatch) {
        auto[nabla_w, nabla_b] = backPropagate(x, y);
        for (int layer = 0; layer < deltaBiases.size(); ++layer) {
            for (int neuron = 0; neuron < deltaBiases[layer].size(); ++neuron) {
                deltaBiases[layer][neuron] += nabla_b[layer][neuron];
            }
            for (int from = 0; from < deltaWeights[layer].size(); ++from) {
                for (int to = 0; to < deltaWeights[layer].size(); ++to) {
                    deltaWeights[layer][from][to] += nabla_w[layer][from][to];
                }
            }
        }
    }

    for (int layer = 0; layer < deltaBiases.size(); ++layer) {
        for (int neuron = 0; neuron < deltaBiases[layer].size(); ++neuron) {
            layerBiases[layer][neuron] -= deltaBiases[layer][neuron] * (learningRate / miniBatch.size());
        }
        for (int from = 0; from < deltaWeights[layer].size(); ++from) {
            for (int to = 0; to < deltaWeights[layer].size(); ++to) {
                layerWeights[layer][from][to] -= deltaWeights[layer][from][to] * (learningRate / miniBatch.size());
            }
        }
    }
}

//TODO implement back propagation mechanisms
std::pair<std::vector<std::vector<std::vector<double>>>, std::vector<std::vector<double>>>
Network::backPropagate(const std::vector<double> &test, int label) {
    return std::pair<std::vector<std::vector<std::vector<double>>>, std::vector<std::vector<double>>>();
}

Network::Network(std::vector<std::pair<std::vector<double>, int>> trainingData,
                 std::vector<std::pair<std::vector<double>, int>> testData,
                 std::vector<int> layerSizes) : trainingData(std::move(trainingData)), testData(std::move(testData)),
                                                layerSizes(std::move(layerSizes)) {
    rGen.seed(time(nullptr));
    distribution = std::uniform_real_distribution<>(0, 1);

    layerBiases.resize(layerSizes.size() - 1);
    for (int layer = 0; layer < (int) layerSizes.size() - 1; ++layer) {
        layerBiases[layer].resize(layerSizes[layer + 1]);
    }

    layerWeights.resize(layerSizes.size() - 1);
    for (int layer = 0; layer < (int) layerSizes.size() - 1; ++layer) {
        layerWeights[layer].resize(layerSizes[layer]);
        for (int layerFrom = 0; layerFrom < (int) layerWeights[layer].size(); ++layerFrom) {
            layerWeights[layer][layerFrom].resize(layerSizes[layer + 1]);
        }
    }
    for (auto &layerBias : layerBiases) {
        for (double &bias : layerBias) {
            bias = distribution(rGen);
        }
    }

    for (auto &layerWeight: layerWeights) {
        for (auto &weights: layerWeight) {
            for (double &weight : weights) {
                weight = distribution(rGen);
            }
        }
    }
}

std::vector<double> Network::dot(const std::vector<std::vector<double>> &m, const std::vector<double> &v) {
    std::vector<double> dot(m.front().size());
    for (int i = 0; i < (int) dot.size(); ++i) {
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
