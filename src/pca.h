#ifndef _PCA_
#define _PCA_

#include <Eigen/Dense>

/* returns covariance matrix of variables being columns of X */
Eigen::MatrixXd getCovarianceMatrix(Eigen::MatrixXd X);

/* Standarize X before using any PCA function! */
/* Functions below return the feature matrix, to recast the data multiply data_matrix * feature_matrix */

/* returns feature matrix with n_columns = components */
Eigen::MatrixXd pcaComponents(Eigen::MatrixXd X, int components);
/* returns feature matrix with enough columns to preserve at least <fraction> of information */
Eigen::MatrixXd pcaFraction(Eigen::MatrixXd X, double fraction);

#endif