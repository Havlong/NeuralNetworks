#include "Network.hpp"

/**
 * Feed-forwards test data into the neural net and determines the amount of correct answers
 *
 * @return Amount of correct answers
 */
int Network::evaluate() {
    int amount = 0;
    for (auto &[X, y]: testData) {
        auto result = feedForward(X);
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

        for (int batchStart = 0; batchStart < trainingData.size(); batchStart += batchSize) {
            if (verbose) {
                while (batchStart / static_cast<double>(trainingData.size()) > lastPercent && lastPercent < 1) {
                    std::cout << lastPercent * 100 << "%..." << std::flush;
                    lastPercent += 0.1;
                }
            }
            int batchEnd = batchStart + batchSize;
            std::vector<std::pair<activation, label>>::const_iterator begin = trainingData.cbegin() + batchStart, end;

            if (batchEnd < trainingData.size())
                end = trainingData.cbegin() + batchEnd;
            else
                end = trainingData.cend();
            trainCorrect += applyMiniBatch(begin, end, learningRate);
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

int Network::applyMiniBatch(std::vector<std::pair<activation, label>>::const_iterator begin,
                            const std::vector<std::pair<activation, label>>::const_iterator &end,
                            const double &learningRate) {
    auto batchSize = static_cast<double>(std::distance(begin, end));

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
    while (begin != end) {
        const auto &[X, y] = *(begin++);

        auto result = feedForward(X);
        auto predictedLabel = (label) (std::max_element(result.begin(), result.end()) - result.begin());
        correctAmount += (predictedLabel == y);

        auto [nabla_w, nabla_b] = backPropagate(X, y);
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
            layerBiases[layer][neuron] -= deltaBiases[layer][neuron] * (learningRate / batchSize);
        }
    }
    for (int layer = 0; layer < deltaWeights.size(); ++layer) {
        for (int from = 0; from < deltaWeights[layer].size(); ++from) {
            for (int to = 0; to < deltaWeights[layer][from].size(); ++to) {
                layerWeights[layer][from][to] -= deltaWeights[layer][from][to] * (learningRate / batchSize);
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
    vec<double> delta = costFunction->delta(layerActivation.back(), trueLabel, layerZ.back());
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
                 std::vector<std::pair<activation, label>> testData, layer<int> layerSizes, Cost *costFunction)
        : trainingData(std::move(trainingData)), testData(std::move(testData)), layerSizes(std::move(layerSizes)),
          costFunction(costFunction) {

    resizeLayers();

    for (auto &biases: layerBiases) {
        for (double &bias: biases) {
            bias = distribution(rGen);
        }
    }

    for (auto &weights: layerWeights) {
        auto c = sqrt(weights.size());
        for (auto &neuronWeights: weights) {
            for (double &weight: neuronWeights) {
                weight = distribution(rGen) / c;
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
