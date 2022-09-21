/**
 * 21.09.2022
 * CrossEntropyCost
 *
 * @author Havlong
 * Copyright (c) 2022 Arbina. All rights reserved.
 */

#ifndef NEURALNETWORKS_CROSSENTROPYCOST_HPP
#define NEURALNETWORKS_CROSSENTROPYCOST_HPP

#include "Cost.hpp"

class CrossEntropyCost : public Cost {
public:
    double fn(vec<double> x, label y) override;

    double delta(double x, double y, double z) override;

};


#endif //NEURALNETWORKS_CROSSENTROPYCOST_HPP
