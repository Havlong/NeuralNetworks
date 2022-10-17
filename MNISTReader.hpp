/**
 * 17.05.2021
 * MNISTReader
 *
 * @author Havlong
 * @version v1.0
 */

#ifndef NEURALNETWORKS_MNISTREADER_HPP
#define NEURALNETWORKS_MNISTREADER_HPP

#include <cstdio>
#include <string>
#include <vector>
#include <sstream>

namespace MNISTReader {
    void prepare();

    std::pair<std::vector<std::pair<std::vector<double>, int>>, std::vector<std::pair<std::vector<double>, int>>>
    readDataSet(const std::string &trainingImagesFile, const std::string &trainingLabelsFile,
                const std::string &testImagesFile, const std::string &testLabelsFile);

    std::vector<std::pair<std::vector<double>, int>>
    readFromFile(const std::string &imagesFile, const std::string &labelsFile);

    std::pair<std::vector<double>, int> getRandomSample(const std::vector<std::pair<std::vector<double>, int>> &data);

    std::string digitToString(const std::pair<std::vector<double>, int> &sample);
}


#endif //NEURALNETWORKS_MNISTREADER_HPP
