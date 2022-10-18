/**
 * 18.10.2022
 * test
 *
 * @author Havlong
 */

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

    auto testData = MNISTReader::readFromFile("../MNIST/t10k-images.idx3-ubyte", "../MNIST/t10k-labels.idx1-ubyte");

    std::unique_ptr<Cost> cost(new CrossEntropyCost());

    Network nn({28 * 28, 128, 10}, cost.get());
    if (fileExists(netFilename)) {
        nn.load(netFilename);
    } else {
        std::cerr << "There was no DNN to evaluate" << std::endl;
        return -1;
    }

    int testCorrect = nn.evaluate(testData);
    double testAccuracy = testCorrect * 100.0 / (int) (testData.size());

    std::cout << "Success rate on testing data is " << testAccuracy << "%\t";
    std::cout << "Correctly predicted: " << testCorrect << " / " << testData.size() << '\n';
    std::cout << std::endl;

    std::cout << "Prediction on random data from test dataset:\n";
    std::pair<activation, label> sample = MNISTReader::getRandomSample(testData);

    activation output = nn.feedForward(sample.first);
    std::cout << "Predicted: " << std::max_element(output.begin(), output.end()) - output.begin() << '\n';
    std::cout << "True Digit was: " << sample.second << std::endl;

    std::cout << MNISTReader::digitToString(sample);

    return 0;
}
