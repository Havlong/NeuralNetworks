#include "Network.hpp"
#include "MNISTReader.hpp"

int main() {
    MNISTReader::prepare();

    auto [trainingData, testData] = MNISTReader::readDataSet("../MNIST/train-images.idx3-ubyte",
                                                             "../MNIST/train-labels.idx1-ubyte",
                                                             "../MNIST/t10k-images.idx3-ubyte",
                                                             "../MNIST/t10k-labels.idx1-ubyte");

    std::unique_ptr<Cost> cost(new CrossEntropyCost());
    Network deepNet({28 * 28, 128, 10}, cost.get());
    deepNet.sgd(trainingData, testData, 30, 10, 0.05, true);

    std::cout << "Prediction on random data from test dataset:\n";

    std::pair<activation, label> sample = MNISTReader::getRandomSample(testData);

    activation output = deepNet.feedForward(sample.first);
    std::cout << "Predicted: " << std::max_element(output.begin(), output.end()) - output.begin() << '\n';
    std::cout << "True Digit was: " << sample.second << std::endl;

    std::cout << MNISTReader::digitToString(sample);

    return 0;
}
