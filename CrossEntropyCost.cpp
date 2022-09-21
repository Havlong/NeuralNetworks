/**
 * 21.09.2022
 * CrossEntropyCost
 *
 * @author Havlong
 * Copyright (c) 2022 Arbina. All rights reserved.
 */

#include "CrossEntropyCost.hpp"
#include <cfloat>

double CrossEntropyCost::fn(vec<double> x, label y) {
    double f = 0;
    for (int i = 0; i < x.size(); ++i) {
        if (x[i] == 0)
            f += (y == i ? static_cast<double>(FLT_MAX) : 0);
        else if (x[i] == 1)
            f += (y == i ? 0 : static_cast<double>(-FLT_MAX));
        else
            f += (y == i ? 0 : 1) * log(1 - x[i]) + (y == i ? -1 : 0) * log(x[i]);
        if (f < -FLT_MAX)
            f = -FLT_MAX;
        if (f > FLT_MAX)
            f = FLT_MAX;
    }
    return f;
}

double CrossEntropyCost::delta(double x, double y, double z) {
    return x - y;
}
