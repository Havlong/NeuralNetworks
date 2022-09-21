/**
 * 21.09.2022
 * QuadraticCost
 *
 * @author Havlong
 * Copyright (c) 2022 Arbina. All rights reserved.
 */

#ifndef NEURALNETWORKS_QUADRATICCOST_HPP
#define NEURALNETWORKS_QUADRATICCOST_HPP

#include "Cost.hpp"

class QuadraticCost : public Cost {
public:
    double fn(vec<double> x, label y) override;

    double delta(double x, double y, double z) override;
};


#endif //NEURALNETWORKS_QUADRATICCOST_HPP
