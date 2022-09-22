#include "Network.hpp"
#include "MNISTReader.hpp"
#include "CrossEntropyCost.hpp"

int main() {

    auto [trainingData, testData] = MNISTReader::readDataSet("../MNIST/train-images.idx3-ubyte",
                                                             "../MNIST/train-labels.idx1-ubyte",
                                                             "../MNIST/t10k-images.idx3-ubyte",
                                                             "../MNIST/t10k-labels.idx1-ubyte");

    CrossEntropyCost costFunction;
    Network deepNet({28 * 28, 128, 10}, &costFunction);
    deepNet.sgd(trainingData, testData, 30, 10, 0.05, true);

    std::cout << "Prediction on random data from test dataset: \n";
    int i = static_cast<int>(random() % testData.size());
    activation output = deepNet.feedForward(testData[i].first);
    std::cout << "Predicted: " << std::max_element(output.begin(), output.end()) - output.begin() << '\n';
    std::cout << "True Digit was: " << testData[i].second << std::endl;

    return 0;
}
