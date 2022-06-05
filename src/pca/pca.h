#ifndef _PCA_
#define _PCA_

#include <Eigen/Dense>

Eigen::MatrixXf getCovarianceMatrix(Eigen::MatrixXf X);
Eigen::MatrixXf pcaComponents(Eigen::MatrixXf X, int components);
Eigen::MatrixXf pcaFraction(Eigen::MatrixXf X, double fraction);

#endif