/**
 * 21.09.2022
 * MathUtils
 *
 * @author Havlong
 */

#ifndef NEURALNETWORKS_MATHUTILS_HPP
#define NEURALNETWORKS_MATHUTILS_HPP

#include <cmath>
#include <vector>

template<typename T> using vec = std::vector<T>;
template<typename T> using mat = vec<vec<T>>;
typedef int label;

vec<double> operator+(const vec<double> &a, const vec<double> &b);

vec<double> operator*(const vec<double> &a, const vec<double> &b);

vec<double> dot(const mat<double> &m, const vec<double> &v);

vec<double> transposedDot(const mat<double> &m, const vec<double> &v);

mat<double> transposedDot(const vec<double> &a, const vec<double> &b);

double sigmoid(const double &z);

vec<double> sigmoid(const vec<double> &z);

double sigmoidPrime(const double &z);

vec<double> sigmoidPrime(const vec<double> &z);

#endif //NEURALNETWORKS_MATHUTILS_HPP
