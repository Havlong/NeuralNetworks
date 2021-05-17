#ifndef NEURALNETWORKS_MNISTREADER_H
#define NEURALNETWORKS_MNISTREADER_H

#include <cstdio>
#include <string>
#include <vector>

/**
 * 17.05.2021
 * MNISTReader
 *
 * @author Havlong
 * @version v1.0
 */
namespace MNISTReader {

    std::pair<std::vector<std::pair<std::vector<double>, int>>, std::vector<std::pair<std::vector<double>, int>>>
    readDataSet(const std::string &trainingImagesFile, const std::string &trainingLabelsFile,
                const std::string &testImagesFile, const std::string &testLabelsFile);

    std::vector<std::pair<std::vector<double>, int>>
    readFromFile(const std::string &imagesFile, const std::string &labelsFile);
}


#endif //NEURALNETWORKS_MNISTREADER_H
