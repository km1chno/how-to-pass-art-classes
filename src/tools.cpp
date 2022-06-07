#include <Eigen/Dense>
#include "tools.h"
#include <iostream>

void standarizeData(Eigen::MatrixXd &A, Eigen::MatrixXd &B, Eigen::MatrixXd &C) {
    Eigen::VectorXd mean = A.colwise().mean();
    Eigen::VectorXd stdev = Eigen::VectorXd::Zero(A.cols());
    for (int i = 0; i < A.rows(); i++)
        for (int j = 0; j < A.cols(); j++) {
            A(i, j) -= mean(j);
            stdev(j) += A(i, j)*A(i, j);
        }
    for (int j = 0; j < A.cols(); j++)
        stdev(j) = std::sqrt(stdev(j)/A.rows());

    for (int i = 0; i < A.rows(); i++)
        for (int j = 0; j < A.cols(); j++)
            A(i, j) /= stdev(j);

    for (int i = 0; i < B.rows(); i++)
        for (int j = 0; j < B.cols(); j++)
            B(i, j) = (B(i, j) - mean(j))/stdev(j);

    for (int i = 0; i < C.rows(); i++)
        for (int j = 0; j < C.cols(); j++)
            C(i, j) = (C(i, j) - mean(j))/stdev(j);
}