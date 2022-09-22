/**
 * 21.09.2022
 * MathUtils
 *
 * @author Havlong
 */

#include "MathUtils.hpp"

/**
 * @return Sum of two vectors
 */
vec<double> operator+(const vec<double> &a, const vec<double> &b) {
    if (a.size() != b.size())
        exit(-1);
    vec<double> result(a.size());
    for (int i = 0; i < a.size(); ++i) {
        result[i] = a[i] + b[i];
    }
    return result;
}

/**
 * @return Hadamard product of two vectors
 */
vec<double> operator*(const vec<double> &a, const vec<double> &b) {
    if (a.size() != b.size())
        exit(-1);
    vec<double> result(a.size());
    for (int i = 0; i < a.size(); ++i) {
        result[i] = a[i] * b[i];
    }
    return result;
}

/**
 * Matrix product of vector and matrix
 * @return Result of operation
 */
vec<double> dot(const mat<double> &m, const vec<double> &v) {
    if (m.size() != v.size())
        exit(-1);
    vec<double> dot(m.front().size());
    for (int i = 0; i < (int) dot.size(); ++i) {
        for (int j = 0; j < (int) v.size(); ++j) {
            dot[i] += m[j][i] * v[j];
        }
    }
    return dot;
}

vec<double> transposedDot(const mat<double> &m, const vec<double> &v) {
    if (m.front().size() != v.size())
        exit(-1);
    vec<double> dot(m.size());
    for (int i = 0; i < (int) m.size(); ++i) {
        for (int j = 0; j < (int) v.size(); ++j) {
            dot[i] += m[i][j] * v[j];
        }
    }
    return dot;
}

mat<double> transposedDot(const vec<double> &a, const vec<double> &b) {
    mat<double> result(a.size(), std::vector<double>(b.size()));
    for (int i = 0; i < a.size(); ++i) {
        for (int j = 0; j < b.size(); ++j) {
            result[i][j] = a[i] * b[j];
        }
    }
    return result;
}

double sigmoid(const double &z) {
    return 1.0 / (1.0 + exp(-z));
}

vec<double> sigmoid(const vec<double> &z) {
    vec<double> result(z.size());
    for (int i = 0; i < result.size(); ++i) {
        result[i] = sigmoid(z[i]);
    }
    return result;
}

double sigmoidPrime(const double &z) {
    return sigmoid(z) * (1.0 - sigmoid(z));
}

vec<double> sigmoidPrime(const vec<double> &z) {
    vec<double> result(z.size());
    for (int i = 0; i < result.size(); ++i) {
        result[i] = sigmoidPrime(z[i]);
    }
    return result;
}
