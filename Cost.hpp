/**
 * 21.09.2022
 * Cost
 *
 * @author Havlong
 * Copyright (c) 2022 Arbina. All rights reserved.
 */

#ifndef NEURALNETWORKS_COST_HPP
#define NEURALNETWORKS_COST_HPP

#include "MathUtils.hpp"

class Cost {
public:
    virtual double fn(vec<double> x, label y) = 0;

    virtual double delta(double x, double y, double z) = 0;

    vec<double> delta(vec<double> x, label y, vec<double> z) {
        vec<double> result(x.size());
        for (int i = 0; i < x.size(); ++i) {
            result[i] = delta(x[i], (y == i ? 1 : 0), z[i]);
        }
        return result;
    }
};


#endif //NEURALNETWORKS_COST_HPP
