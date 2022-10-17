/**
 * 17.05.2021
 * MNISTReader
 *
 * @author Havlong
 * @version v1.0
 */

#include "MNISTReader.hpp"

void reverseInt(uint32_t &source) {
    uint8_t b1, b2, b3, b4;
    b1 = source & 255;
    b2 = (source >> 8) & 255;
    b3 = (source >> 16) & 255;
    b4 = (source >> 24) & 255;

    source = ((uint32_t) b1 << 24) | ((uint32_t) b2 << 16) | ((uint32_t) b3 << 8) | (uint32_t) b4;
}

std::vector<std::pair<std::vector<double>, int>>
MNISTReader::readFromFile(const std::string &imagesFile, const std::string &labelsFile) {
    FILE *imageFP = fopen(imagesFile.c_str(), "rb");
    uint32_t magic = 0, size = 0, rows = 0, columns = 0;
    fread(&magic, sizeof(magic), 1, imageFP);
    fread(&size, sizeof(size), 1, imageFP);
    fread(&rows, sizeof(rows), 1, imageFP);
    fread(&columns, sizeof(columns), 1, imageFP);
    reverseInt(magic);
    reverseInt(size);
    reverseInt(rows);
    reverseInt(columns);
    if (magic != 2051)
        return {};
    std::vector<std::pair<std::vector<double>, int>> data(size, std::pair(std::vector<double>(rows * columns), 0));

    for (auto &[image, _]: data) {
        for (int i = 0; i < rows * columns; ++i) {
            uint8_t x;
            fread(&x, sizeof(x), 1, imageFP);
            image[i] = x / 255.0;
        }
    }
    fclose(imageFP);

    FILE *labelFP = fopen(labelsFile.c_str(), "rb");
    uint32_t labelSize;
    fread(&magic, sizeof(magic), 1, labelFP);
    fread(&labelSize, sizeof(labelSize), 1, labelFP);
    reverseInt(magic);
    reverseInt(labelSize);
    if (magic != 2049 || labelSize != size)
        return {};

    for (auto &[_, label]: data) {
        uint8_t x;
        fread(&x, sizeof(x), 1, labelFP);
        label = x;
    }
    fclose(labelFP);
    return data;
}

std::pair<std::vector<std::pair<std::vector<double>, int>>, std::vector<std::pair<std::vector<double>, int>>>
MNISTReader::readDataSet(const std::string &trainingImagesFile, const std::string &trainingLabelsFile,
                         const std::string &testImagesFile, const std::string &testLabelsFile) {
    return {readFromFile(trainingImagesFile, trainingLabelsFile), readFromFile(testImagesFile, testLabelsFile)};
}

std::pair<std::vector<double>, int>
MNISTReader::getRandomSample(const std::vector<std::pair<std::vector<double>, int>> &data) {
    int i = static_cast<int>(random() % data.size());
    return data[i];
}

void MNISTReader::prepare() {
    srandom(time(nullptr));
}

std::string MNISTReader::digitToString(const std::pair<std::vector<double>, int> &sample) {
    std::stringstream stringBuffer;
    const int rows = 28, columns = 28;

    for (int i = 0; i < columns + 2; ++i) {
        stringBuffer << '_';
    }
    stringBuffer << '\n';
    for (int i = 0; i < rows; ++i) {
        stringBuffer << '|';
        for (int j = 0; j < columns; ++j) {
            if (sample.first[i * columns + j] < 0.3)
                stringBuffer << ' ';
            else
                stringBuffer << '*';
        }
        stringBuffer << "|\n";
    }
    stringBuffer << '|';
    for (int i = 0; i < 28; ++i) {
        stringBuffer << '_';
    }
    stringBuffer << '|' << std::endl;
    return stringBuffer.str();
}
