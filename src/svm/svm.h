#ifndef _LIN_SVM_
#define _LIN_SVM_

#include <vector>
#include "Dense"

class LinSVM {
    int m;                      /* number of observations in training set */
    int d;                      /* dimension of observations */
    Eigen::MatrixXd Xt, Xv;     /* training, verification set */
    Eigen::VectorXd yt, yv;     /* training, verification classes */
    std::vector<double> C;      /* vector of considered parameters C */

    Eigen::VectorXd w;          /* w^Tx + b is the resulting hyperplane */
    double b;

    Eigen::VectorXd a;          /* lagrange multipliers in dual problem */
    Eigen::MatrixXd K;          /* gram matrix */

    /* returns accuracy on verification set using current w, b */
    double calc_accuracy();
    /* sequential minimal optimization with parameter c */
    void SMO(double c);

public:
    LinSVM(int _d, Eigen::MatrixXd _Xt, Eigen::MatrixXd _Xv, Eigen::VectorXd _yt, Eigen::VectorXd _yv);
    /* classifies x with hyperplane w, b */
    int classify(const Eigen::VectorXd &x);
    /* performs the whole learning process */
    /* considers n_c values of parameter c equally distributed over range [low_c, high_c] */
    void fit();
};

#endif