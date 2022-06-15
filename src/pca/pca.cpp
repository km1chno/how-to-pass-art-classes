#include <Eigen/Dense>
#include <iostream>
#include "pca.h"

Eigen::MatrixXd getCovarianceMatrix(Eigen::MatrixXd X) {
    Eigen::MatrixXd centered = X.rowwise() - X.colwise().mean();
    Eigen::MatrixXd cov = (centered.adjoint() * centered) / double(X.rows() - 1);
    return cov;
}

Eigen::MatrixXd pcaComponents(Eigen::MatrixXd X, int components) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(getCovarianceMatrix(X));
    return es.eigenvectors().rightCols(components);
}

Eigen::MatrixXd pcaFraction(Eigen::MatrixXd X, double fraction) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(getCovarianceMatrix(std::move(X)));
    double S = es.eigenvalues().sum();
    double eigenSum = 0;
    int components = 0;
    for (int i = es.eigenvalues().size()-1; i >= 0; i--) {
        eigenSum += es.eigenvalues()(i);
        components++;
        if (eigenSum/S >= fraction)
            break;
    }
    return es.eigenvectors().rightCols(components);
}