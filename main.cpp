#include "Network.hpp"
#include "MNISTReader.hpp"
#include "CrossEntropyCost.hpp"

int main() {

    auto [trainingData, testData] = MNISTReader::readDataSet("../train-images.idx3-ubyte", "../train-labels.idx1-ubyte",
                                                             "../t10k-images.idx3-ubyte", "../t10k-labels.idx1-ubyte");

    CrossEntropyCost costFunction;
    Network deepNet(trainingData, testData, {784, 100, 10}, &costFunction);
    deepNet.sgd(250, 10, 0.05, false);

    return 0;
}
