/**
 * 21.09.2022
 * QuadraticCost
 *
 * @author Havlong
 * Copyright (c) 2022 Arbina. All rights reserved.
 */

#include "QuadraticCost.hpp"

double QuadraticCost::fn(vec<double> x, label y) {
    double f = 0;
    for (int i = 0; i < x.size(); ++i) {
        if (y != i)
            f += x[i] * x[i];
        else
            f += (1 - x[i]) * (1 - x[i]);
    }
    return f / 2;
}

double QuadraticCost::delta(double x, double y, double z) {
    return (x - y) * sigmoidPrime(z);
}
