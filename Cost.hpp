/**
 * 21.09.2022
 * Cost
 *
 * @author Havlong
 */

#ifndef NEURALNETWORKS_COST_HPP
#define NEURALNETWORKS_COST_HPP

#include "MathUtils.hpp"

class Cost {
public:
    virtual double fn(vec<double> x, label y) = 0;

    virtual double delta(double x, double y, double z) = 0;

    vec<double> delta(vec<double> x, label y, vec<double> z);
};

class QuadraticCost : public Cost {
public:
    double fn(vec<double> x, label y) override;

    double delta(double x, double y, double z) override;
};

class CrossEntropyCost : public Cost {
public:
    double fn(vec<double> x, label y) override;

    double delta(double x, double y, double z) override;

};

#endif //NEURALNETWORKS_COST_HPP
