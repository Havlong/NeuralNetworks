#include "Network.h"
#include "MNISTReader.h"

int main() {

    auto [trainingData, testData] = MNISTReader::readDataSet("../train-images.idx3-ubyte", "../train-labels.idx1-ubyte",
                                                             "../t10k-images.idx3-ubyte", "../t10k-labels.idx1-ubyte");

    Network deepNet(trainingData, testData, {784, 100, 30, 10});
    deepNet.sgd(40, 10, 0.01);
/*
    std::vector<double> one = {0.0, 0.0, 1.0, 0.0, 0.0,
                               0.0, 0.0, 1.0, 0.0, 0.0,
                               0.0, 0.0, 1.0, 0.0, 0.0,
                               0.0, 0.0, 1.0, 0.0, 0.0,
                               0.0, 0.0, 1.0, 0.0, 0.0};
    std::vector<double> zero = {0.0, 0.0, 1.0, 0.0, 0.0,
                                0.0, 1.0, 0.0, 1.0, 0.0,
                                1.0, 0.0, 0.0, 0.0, 1.0,
                                0.0, 1.0, 0.0, 1.0, 0.0,
                                0.0, 0.0, 1.0, 0.0, 0.0};
    std::vector<std::pair<std::vector<double>, int>> trainingData;
    std::vector<std::pair<std::vector<double>, int>> testingData;

    std::pair<std::vector<double>, int> t0 = {zero, 0}, t1 = {one, 1};

    for (int i = 0; i < 1000; ++i) {
        trainingData.push_back(t0);
        trainingData.push_back(t1);
    }

    for (int i = 0; i < rand() % 500 + 750; ++i) {
        testingData.push_back((rand() % 2 ? t1 : t0));
    }

    Network testNet(trainingData, testingData, {25, 5, 2});
    testNet.sgd(30, 10, 0.01);
*/

    return 0;
}