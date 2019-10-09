#include "Network.h"

int main() {
    Network myNet(std::vector<std::pair<std::vector<double>, int>>(),
                  std::vector<std::pair<std::vector<double>, int>>(), {784, 30, 10});
    return 0;
}