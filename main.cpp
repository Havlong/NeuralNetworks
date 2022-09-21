#include "Network.hpp"
#include "MNISTReader.hpp"
#include "QuadraticCost.hpp"

int main() {

    auto [trainingData, testData] = MNISTReader::readDataSet("../MNIST/train-images.idx3-ubyte",
                                                             "../MNIST/train-labels.idx1-ubyte",
                                                             "../MNIST/t10k-images.idx3-ubyte",
                                                             "../MNIST/t10k-labels.idx1-ubyte");

    QuadraticCost costFunction;
    Network deepNet(trainingData, testData, {28 * 28, 128, 10}, &costFunction);
    deepNet.sgd(5, 10, 0.05, true);

    return 0;
}
