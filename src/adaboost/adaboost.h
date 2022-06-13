#ifndef _ADABOOST_
#define _ADABOOST_

#include <vector>
#include <functional>
#include "Dense"

class AdaBoost {
    int m;                  /* number of observations in training set */
    int d;                  /* dimension of points */
    /* ASSUMPTION: features are scaled into [0, 1] range */
    Eigen::MatrixXd X;      /* training data */
    /* ASSUMPTION: labels are -1 or 1 */
    Eigen::VectorXd y;      /* training labels */

    int n_stumps;           /* number of random stumps to use */
    int n_steps;            /* number of iterations of main loop */

    Eigen::VectorXd a;      /* weights of stumps */
    Eigen::VectorXd w;      /* weights of observations */

    std::vector<std::function<int(Eigen::VectorXd)>> stumps;    /* well... the stumps */

public:
    AdaBoost(int _d, Eigen::MatrixXd _X, Eigen::MatrixXd _y, int _n_stumps, int _n_steps);
    /* calculates weighted accuracy of stump i on training data */
    double calcStumpAccuracy(int i);
    /* generated random decision stumps */
    void prepareStumps();
    /* returns classification for x, -1 or 1 */
    int classify(const Eigen::VectorXd &x);
    /* performs the learning process */
    void fit();
};

#endif