#include "Network.hpp"
#include "MNISTReader.hpp"
#include <sys/stat.h>

inline bool fileExists(const std::string &filename) {
    struct stat buffer{};
    return (stat(filename.c_str(), &buffer) == 0);
}

const std::string netFilename = "../network.dnn";

int main() {
    MNISTReader::prepare();

    auto [trainingData, testData] = MNISTReader::readDataSet("../MNIST/train-images.idx3-ubyte",
                                                             "../MNIST/train-labels.idx1-ubyte",
                                                             "../MNIST/t10k-images.idx3-ubyte",
                                                             "../MNIST/t10k-labels.idx1-ubyte");

    std::unique_ptr<Cost> cost(new CrossEntropyCost());

    Network nn({28 * 28, 128, 10}, cost.get());
    if (fileExists(netFilename))
        nn.load(netFilename);

    nn.sgd(trainingData, testData, 30, 10, 0.05, true);
    nn.save(netFilename);

    return 0;
}
