#include "Network.hpp"
#include "MNISTReader.hpp"

void printDigit(const std::vector<double> &digit) {
    for (int i = 0; i < 30; ++i) {
        std::cout << '_';
    }
    std::cout << '\n';
    for (int i = 0; i < 28; ++i) {
        std::cout << '|';
        for (int j = 0; j < 28; ++j) {
            if (digit[i * 28 + j] < 0.3)
                std::cout << ' ';
            else
                std::cout << '*';
        }
        std::cout << "|\n";
    }
    std::cout << '|';
    for (int i = 0; i < 28; ++i) {
        std::cout << '_';
    }
    std::cout << '|' << std::endl;
}

int main() {

    auto [trainingData, testData] = MNISTReader::readDataSet("../MNIST/train-images.idx3-ubyte",
                                                             "../MNIST/train-labels.idx1-ubyte",
                                                             "../MNIST/t10k-images.idx3-ubyte",
                                                             "../MNIST/t10k-labels.idx1-ubyte");

    std::unique_ptr<Cost> cost(new CrossEntropyCost());
    Network deepNet({28 * 28, 128, 10}, cost.get());
    deepNet.sgd(trainingData, testData, 30, 10, 0.05, true);

    std::cout << "Prediction on random data from test dataset:\n";

    srandom(time(nullptr));
    int i = static_cast<int>(random() % testData.size());

    activation output = deepNet.feedForward(testData[i].first);
    std::cout << "Predicted: " << std::max_element(output.begin(), output.end()) - output.begin() << '\n';
    std::cout << "True Digit was: " << testData[i].second << std::endl << std::endl;

    printDigit(testData[i].first);

    return 0;
}
