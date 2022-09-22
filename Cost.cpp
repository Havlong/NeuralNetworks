/**
 * 22.09.2022
 * Cost
 *
 * @author Havlong
 */
#include "Cost.hpp"

#include <cfloat>

vec<double> Cost::delta(vec<double> x, label y, vec<double> z) {
    vec<double> result(x.size());
    for (int i = 0; i < x.size(); ++i) {
        result[i] = delta(x[i], (y == i ? 1 : 0), z[i]);
    }
    return result;
}

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
