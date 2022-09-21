#include "Network.h"

/**
 * @return Sum of two vectors
 */
vec<double> operator+(const vec<double> &a, const vec<double> &b) {
    if (a.size() != b.size())
        exit(-1);
    vec<double> result(a.size());
    for (int i = 0; i < a.size(); ++i) {
        result[i] = a[i] + b[i];
    }
    return result;
}

/**
 * @return Hadamard product of two vectors
 */
vec<double> operator*(const vec<double> &a, const vec<double> &b) {
    if (a.size() != b.size())
        exit(-1);
    vec<double> result(a.size());
    for (int i = 0; i < a.size(); ++i) {
        result[i] = a[i] * b[i];
    }
    return result;
}

/**
 * Matrix product of vector and matrix
 * @return Result of operation
 */
vec<double> dot(const mat<double> &m, const vec<double> &v) {
    if (m.size() != v.size())
        exit(-1);
    vec<double> dot(m.front().size());
    for (int i = 0; i < (int) dot.size(); ++i) {
        for (int j = 0; j < (int) v.size(); ++j) {
            dot[i] += m[j][i] * v[j];
        }
    }
    return dot;
}

vec<double> transposedDot(const mat<double> &m, const vec<double> &v) {
    if (m.front().size() != v.size())
        exit(-1);
    vec<double> dot(m.size());
    for (int i = 0; i < (int) m.size(); ++i) {
        for (int j = 0; j < (int) v.size(); ++j) {
            dot[i] += m[i][j] * v[j];
        }
    }
    return dot;
}

mat<double> transposedDot(const vec<double> &a, const vec<double> &b) {
    mat<double> result(a.size(), std::vector<double>(b.size()));
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
 * @return Amount of correct answers
 */
int Network::evaluate() {
    int amount = 0;
    for (auto &[x, y]: testData) {
        auto result = feedForward(x);
        auto resultLabel = (label) (std::max_element(result.begin(), result.end()) - result.begin());
        amount += (resultLabel == y);
    }
    return amount;
}

void Network::sgd(const int &epochsCount, const int &batchSize, const double &learningRate, bool verbose) {
    std::cout << std::fixed << std::setprecision(2);
    for (int epoch = 1; epoch <= epochsCount; ++epoch) {
        std::shuffle(trainingData.begin(), trainingData.end(), rGen);
        int trainCorrect = 0;

        double lastPercent = 0.1;
        std::cout << "Epoch #" << epoch << " starts\n";

        std::vector<std::pair<activation, label>> batch;
        for (int batchStart = 0; batchStart < trainingData.size(); batchStart += batchSize) {
            if (verbose) {
                while (batchStart / static_cast<double>(trainingData.size()) > lastPercent && lastPercent < 1) {
                    std::cout << lastPercent * 100 << "%..." << std::flush;
                    lastPercent += 0.1;
                }
            }

            for (int i = batchStart; i < trainingData.size() && i < batchStart + batchSize; ++i) {
                batch.emplace_back(trainingData[i]);
            }
            trainCorrect += applyMiniBatch(batch, learningRate);
            batch.clear();
        }
        if (verbose) std::cout << std::endl;

        int testCorrect = evaluate();

        std::cout << "Epoch #" << epoch << " is completed\n";
        std::cout << "Success rate on training data is " << trainCorrect * 100.0 / (int) (trainingData.size()) << "%\t";
        if (verbose) std::cout << "Correctly predicted: " << trainCorrect << " / " << trainingData.size() << '\n';
        std::cout << "Success rate on testing data is " << testCorrect * 100.0 / (int) (testData.size()) << "%\t";
        if (verbose) std::cout << "Correctly predicted: " << testCorrect << " / " << testData.size() << '\n';
        std::cout << std::endl;
    }
}

activation Network::feedForward(const activation &input) {
    activation currentActivation = input;
    for (int layer = 0; layer < (int) layerWeights.size(); ++layer) {
        currentActivation = dot(layerWeights[layer], currentActivation) + layerBiases[layer];
    }
    return currentActivation;
}

double Network::sigmoid(const double &z) {
    return 1.0 / (1.0 + exp(-z));
}

vec<double> Network::sigmoid(const vec<double> &z) {
    vec<double> result(z.size());
    for (int i = 0; i < result.size(); ++i) {
        result[i] = sigmoid(z[i]);
    }
    return result;
}

double Network::sigmoidPrime(const double &z) {
    return sigmoid(z) * (1.0 - sigmoid(z));
}

vec<double> Network::sigmoidPrime(const vec<double> &z) {
    vec<double> result(z.size());
    for (int i = 0; i < result.size(); ++i) {
        result[i] = sigmoidPrime(z[i]);
    }
    return result;
}

vec<double> Network::costDerivative(const activation &output, const label &trueLabel) {
    activation result(output.size());
    for (int i = 0; i < output.size(); ++i) {
        result[i] = (trueLabel == i ? -1 : 0) + output[i];
    }
    return result;
}

int Network::applyMiniBatch(const std::vector<std::pair<activation, label>> &miniBatch, const double &learningRate) {
    // Initialize biases shape
    layer<biases> deltaBiases;
    for (auto &biases: layerBiases) {
        deltaBiases.emplace_back(biases.size(), 0.0);
    }

    // Initialize weights shape
    layer<weights> deltaWeights;
    for (auto &weights: layerWeights) {
        deltaWeights.emplace_back();
        for (auto &neuronWeights: weights) {
            deltaWeights.back().emplace_back(neuronWeights.size(), 0.0);
        }
    }

    int correctAmount = 0;
    for (const auto &[x, y]: miniBatch) {
        auto result = feedForward(x);
        auto predictedLabel = (label) (std::max_element(result.begin(), result.end()) - result.begin());
        correctAmount += (predictedLabel == y);

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
            layerBiases[layer][neuron] -=
                    deltaBiases[layer][neuron] * (learningRate / static_cast<int>(miniBatch.size()));
        }
    }
    for (int layer = 0; layer < deltaWeights.size(); ++layer) {
        for (int from = 0; from < deltaWeights[layer].size(); ++from) {
            for (int to = 0; to < deltaWeights[layer][from].size(); ++to) {
                layerWeights[layer][from][to] -=
                        deltaWeights[layer][from][to] * (learningRate / static_cast<int>(miniBatch.size()));
            }
        }
    }
    return correctAmount;
}

std::pair<layer<weights>, layer<biases>> Network::backPropagate(const activation &input, label trueLabel) {
    // Initialize biases shape
    layer<biases> nabla_b;
    for (auto &biases: layerBiases) {
        nabla_b.emplace_back(biases.size(), 0.0);
    }

    // Initialize weights shape
    layer<weights> nabla_w;
    for (auto &weights: layerWeights) {
        nabla_w.emplace_back();
        for (auto &neuronWeights: weights) {
            nabla_w.back().emplace_back(neuronWeights.size(), 0.0);
        }
    }

    // Feed-forward
    activation currentActivation = input;
    std::vector<activation> layerActivation = {currentActivation};
    std::vector<activation> layerZ;
    for (int layer = 0; layer < layerWeights.size(); ++layer) {
        currentActivation = dot(layerWeights[layer], currentActivation) + layerBiases[layer];
        layerZ.emplace_back(currentActivation);

        currentActivation = sigmoid(currentActivation);
        layerActivation.emplace_back(currentActivation);
    }

    // Backwards pass
    vec<double> delta = costDerivative(layerActivation.back(), trueLabel) * sigmoidPrime(layerZ.back());
    nabla_b.back() = delta;
    nabla_w.back() = transposedDot(layerActivation[layerActivation.size() - 2], delta);

    for (int layer = 2; layer < layerSizes.size(); ++layer) {
        auto z = layerZ[layerZ.size() - layer];
        delta = transposedDot(layerWeights[layerWeights.size() - layer + 1], delta) * sigmoidPrime(z);
        nabla_b[nabla_b.size() - layer] = delta;
        nabla_w[nabla_w.size() - layer] = transposedDot(layerActivation[layerActivation.size() - layer - 1], delta);
    }

    return {nabla_w, nabla_b};
}

Network::Network(std::vector<std::pair<activation, label>> trainingData,
                 std::vector<std::pair<activation, label>> testData, layer<int> layerSizes) : trainingData(
        std::move(trainingData)), testData(std::move(testData)), layerSizes(std::move(layerSizes)) {

    resizeLayers();

    for (auto &biases: layerBiases) {
        for (double &bias: biases) {
            bias = distribution(rGen);
        }
    }

    for (auto &weights: layerWeights) {
        for (auto &neuronWeights: weights) {
            for (double &weight: neuronWeights) {
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
