#ifndef _TOOLS_
#define _TOOLS_

#include <Eigen/Dense>

/* standarizes data in all three matrices using means and variances from A */
void standarizeData(Eigen::MatrixXd &A, Eigen::MatrixXd &B, Eigen::MatrixXd &C);

/* scales data in all three matrices into range [0, 1] using min max method on matrix A */
void minmaxscaling(Eigen::MatrixXd &A, Eigen::MatrixXd &B, Eigen::MatrixXd &C);

#endif