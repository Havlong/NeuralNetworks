#include "Network.h"
#include "MNISTReader.h"

int main() {
    auto[trainingData, testData] = MNISTReader::readDataSet("train-images.idx3-ubyte", "train-labels.idx1-ubyte",
                                                            "t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte");

    Network myNet(trainingData, testData, {784, 30, 10});

    myNet.sgd(30, 30, 3.0);
    return 0;
}