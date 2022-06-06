#include <Eigen/Dense>
#include "pca.h"

/* returns covariance matrix of variables being columns of X */
Eigen::MatrixXf getCovarianceMatrix(Eigen::MatrixXf X) {
    Eigen::MatrixXf centered = X.rowwise() - X.colwise().mean();
    Eigen::MatrixXf cov = (centered.adjoint() * centered) / double(X.rows() - 1);
    return cov;
}

/* Standarize X before using any PCA function! */
/* Functions below return the feature matrix, to recast the data multiply data_matrix * feature_matrix */

/* returns feature matrix with n_columns = components */
Eigen::MatrixXf pcaComponents(Eigen::MatrixXf X, int components) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> es(getCovarianceMatrix(X));
    return es.eigenvectors().rightCols(components);
}

/* returns feature matrix with enough columns to preserve at least <fraction> of information */
Eigen::MatrixXf pcaFraction(Eigen::MatrixXf X, double fraction) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> es(getCovarianceMatrix(X));
    Eigen::VectorXf normalizedEigenvalues = es.eigenvalues() / es.eigenvalues().sum();

    double eigenSum = 0;
    int components = 0;
    for (int i = normalizedEigenvalues.cols()-1; i >= 0; i--) {
        eigenSum += normalizedEigenvalues(i);
        components++;
        if (eigenSum >= fraction)
            break;
    }
    return es.eigenvectors().rightCols(components);
}

int main() {
    
}