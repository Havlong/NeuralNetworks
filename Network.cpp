#include "Network.h"

/**
 * @return Sum of two vectors
 */
std::vector<double> operator+(const std::vector<double> &a, const std::vector<double> &b) {
    if (a.size() != b.size())
        exit(-1);
    std::vector<double> result(a.size());
    for (int i = 0; i < a.size(); ++i) {
        result[i] = a[i] + b[i];
    }
    return result;
}

/**
 * @return Hadamard product of two vectors
 */
std::vector<double> operator*(const std::vector<double> &a, const std::vector<double> &b) {
    if (a.size() != b.size())
        exit(-1);
    std::vector<double> result(a.size());
    for (int i = 0; i < a.size(); ++i) {
        result[i] = a[i] * b[i];
    }
    return result;
}

/**
 * Matrix product of vector and matrix
 * @return Result of operation
 */
std::vector<double> dot(const std::vector<std::vector<double>> &m, const std::vector<double> &v) {
    if (m.size() != v.size())
        exit(-1);
    std::vector<double> dot(m.front().size());
    for (int i = 0; i < (int) dot.size(); ++i) {
        for (int j = 0; j < (int) v.size(); ++j) {
            dot[i] += m[j][i] * v[j];
        }
    }
    return dot;
}

std::vector<double> transposedDot(const std::vector<std::vector<double>> &m, const std::vector<double> &v) {
    if (m.front().size() != v.size())
        exit(-1);
    std::vector<double> dot(m.size());
    for (int i = 0; i < (int) m.size(); ++i) {
        for (int j = 0; j < (int) v.size(); ++j) {
            dot[i] += m[i][j] * v[j];
        }
    }
    return dot;
}

std::vector<std::vector<double>> transposedDot(const std::vector<double> &a, const std::vector<double> &b) {
    std::vector<std::vector<double>> result(a.size(), std::vector<double>(b.size()));
    for (int i = 0; i < a.size(); ++i) {
        for (int j = 0; j < b.size(); ++j) {
            result[i][j] = a[i] * b[j];
        }
    }
    return result;
}

/**
 * Feed-forwards test data into the neural net and determines the amount of correct answers
 *
 * @return Percentage of correct answers
 */
double Network::evaluate() {
    int amount = 0;
    for (auto &[x, y]: testData) {
        auto result = feedForward(x);
        int resultLabel = (int) (std::max_element(result.begin(), result.end()) - result.begin());
        amount += (resultLabel == y);
    }
    return amount * 100 / ((double) testData.size());
}

void Network::sgd(const int &epochsCount, const int &batchSize, const double &learningRate) {
    for (int epoch = 1; epoch <= epochsCount; ++epoch) {
//        std::shuffle(trainingData.begin(), trainingData.end(), rGen);
        int correctAmount = 0;
        std::vector<std::pair<std::vector<double>, int>> batch;
        for (int batchStart = 0; batchStart < trainingData.size(); batchStart += batchSize) {
            for (int i = batchStart; i < trainingData.size() && i < batchStart + batchSize; ++i) {
                batch.push_back(trainingData[i]);
            }
            correctAmount += applyMiniBatch(batch, learningRate);
            batch.clear();
        }

        double successRate = evaluate();
        std::cout << "Epoch #" << epoch << " is completed\n";
        std::cout << "Success rate on training data is " << correctAmount * 100 / ((double) trainingData.size()) << '%'
                  << std::endl;
        std::cout << "Success rate on testing data is " << successRate << "%\n" << std::endl;
    }
}

std::vector<double> Network::feedForward(const std::vector<double> &input) {
    std::vector<double> currentActivation(input);
    for (int layer = 0; layer < (int) layerWeights.size(); ++layer) {
        auto z = dot(layerWeights[layer], currentActivation) + layerBiases[layer];
        currentActivation.clear();
        currentActivation = z;
    }
    return currentActivation;
}

double Network::sigmoid(const double &z) {
    return 1.0 / (1.0 + exp(-z));
}

std::vector<double> Network::sigmoid(const std::vector<double> &z) {
    std::vector<double> result(z.size());
    for (int i = 0; i < result.size(); ++i) {
        result[i] = sigmoid(z[i]);
    }
    return result;
}

double Network::sigmoidPrime(const double &z) {
    return sigmoid(z) * (1.0 - sigmoid(z));
}

std::vector<double> Network::sigmoidPrime(const std::vector<double> &z) {
    std::vector<double> result(z.size());
    for (int i = 0; i < result.size(); ++i) {
        result[i] = sigmoidPrime(z[i]);
    }
    return result;
}

std::vector<double> Network::costDerivative(const std::vector<double> &output, const int &label) {
    std::vector<double> result(output.size());
    for (int i = 0; i < output.size(); ++i) {
        result[i] = (label == i ? -1 : 0) + output[i];
    }
    return result;
}

int
Network::applyMiniBatch(const std::vector<std::pair<std::vector<double>, int>> &miniBatch, const double &learningRate) {
    // Initialize biases shape
    std::vector<std::vector<double>> deltaBiases(layerBiases.size());
    for (int i = 0; i < deltaBiases.size(); ++i) {
        deltaBiases[i] = std::vector<double>(layerBiases[i].size(), 0.0);
    }

    // Initialize weights shape
    std::vector<std::vector<std::vector<double>>> deltaWeights(layerWeights.size());
    for (int i = 0; i < deltaWeights.size(); ++i) {
        deltaWeights[i] = std::vector<std::vector<double>>(layerWeights[i].size());
        for (int j = 0; j < deltaWeights[i].size(); ++j) {
            deltaWeights[i][j] = std::vector<double>(layerWeights[i][j].size(), 0.0);
        }
    }

    int correctAmount = 0;
    for (const auto &[x, y]: miniBatch) {
        auto result = feedForward(x);
        int resultLabel = (int) (std::max_element(result.begin(), result.end()) - result.begin());
        correctAmount += (resultLabel == y);

        auto [nabla_w, nabla_b] = backPropagate(x, y);
        for (int layer = 0; layer < deltaBiases.size(); ++layer) {
            for (int neuron = 0; neuron < deltaBiases[layer].size(); ++neuron) {
                deltaBiases[layer][neuron] += nabla_b[layer][neuron];
            }
        }
        for (int layer = 0; layer < deltaWeights.size(); ++layer) {
            for (int from = 0; from < deltaWeights[layer].size(); ++from) {
                for (int to = 0; to < deltaWeights[layer][from].size(); ++to) {
                    deltaWeights[layer][from][to] += nabla_w[layer][from][to];
                }
            }
        }
    }

    for (int layer = 0; layer < deltaBiases.size(); ++layer) {
        for (int neuron = 0; neuron < deltaBiases[layer].size(); ++neuron) {
            layerBiases[layer][neuron] -= deltaBiases[layer][neuron] * (learningRate / ((double) miniBatch.size()));
        }
    }
    for (int layer = 0; layer < deltaWeights.size(); ++layer) {
        for (int from = 0; from < deltaWeights[layer].size(); ++from) {
            for (int to = 0; to < deltaWeights[layer][from].size(); ++to) {
                layerWeights[layer][from][to] -=
                        deltaWeights[layer][from][to] * (learningRate / ((double) miniBatch.size()));
            }
        }
    }
    return correctAmount;
}

std::pair<std::vector<std::vector<std::vector<double>>>, std::vector<std::vector<double>>>
Network::backPropagate(const std::vector<double> &test, int label) {
    // Initialize biases shape
    std::vector<std::vector<double>> nabla_b(layerBiases.size());
    for (int i = 0; i < nabla_b.size(); ++i) {
        nabla_b[i] = std::vector<double>(layerBiases[i].size());
    }

    // Initialize weights shape
    std::vector<std::vector<std::vector<double>>> nabla_w(layerWeights.size());
    for (int i = 0; i < nabla_w.size(); ++i) {
        nabla_w[i] = std::vector<std::vector<double>>(layerWeights[i].size());
        for (int j = 0; j < nabla_w[i].size(); ++j) {
            nabla_w[i][j] = std::vector<double>(layerWeights[i][j].size());
        }
    }

    // Feed-forward
    std::vector<double> currentActivation = std::vector<double>(test);
    std::vector<std::vector<double>> activations = {currentActivation};
    std::vector<std::vector<double>> layerFunction;
    for (int layer = 0; layer < layerWeights.size(); ++layer) {
        auto z = dot(layerWeights[layer], currentActivation) + layerBiases[layer];
        layerFunction.push_back(z);
        currentActivation.clear();
        currentActivation = sigmoid(z);
        activations.push_back(currentActivation);
    }

    // Backwards pass
    std::vector<double> delta = costDerivative(activations.back(), label) * sigmoidPrime(layerFunction.back());
    nabla_b.back() = delta;
    nabla_w.back() = transposedDot(activations[activations.size() - 2], delta);

    for (int layer = 2; layer < layerSizes.size(); ++layer) {
        auto z = layerFunction[layerFunction.size() - layer];
        delta = transposedDot(layerWeights[layerWeights.size() - layer + 1], delta) * sigmoidPrime(z);
        nabla_b[nabla_b.size() - layer] = delta;
        nabla_w[nabla_w.size() - layer] = transposedDot(activations[activations.size() - layer - 1], delta);
    }

    return {nabla_w, nabla_b};
}

Network::Network(std::vector<std::pair<std::vector<double>, int>> trainingData,
                 std::vector<std::pair<std::vector<double>, int>> testData, std::vector<int> layerSizes) : trainingData(
        std::move(trainingData)), testData(std::move(testData)), layerSizes(std::move(layerSizes)) {
    resizeLayers();

    rGen.seed(time(nullptr));
    distribution = std::normal_distribution<>();

    for (auto &layer: layerBiases) {
        for (double &bias: layer) {
            bias = distribution(rGen);
        }
    }

    for (auto &layer: layerWeights) {
        for (auto &weights: layer) {
            for (double &weight: weights) {
                weight = distribution(rGen);
            }
        }
    }
}

void Network::resizeLayers() {
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
}

