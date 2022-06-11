#include <vector>
#include <math.h>
#include <iostream>
#include <random>
#include "Dense"

#include "adaboost.h"

AdaBoost::AdaBoost(int _d, Eigen::MatrixXd _X, Eigen::MatrixXd _y, int _n_stumps, int _n_steps) :
    m(_X.rows()), d(_d), X(_X), y(_y), n_stumps(_n_stumps), n_steps(_n_steps) {
    a = Eigen::VectorXd::Zero(n_stumps);
    w = Eigen::VectorXd::Constant(m, 1.0/double(m));
}

void AdaBoost::prepareStumps() {
    /* stump is a function (vector x) -> 1 if x(i) <= A else -1 */
    /* i is a random integer from uniform[0, d-1] */
    /* A is a random double from uniform[0, 1] */
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> real_dist(0.0, 1.0);
    std::uniform_int_distribution<> int_dist(0, d-1);

    for (int k = 0; k < n_stumps; k++) {
        int i = int_dist(gen);
        double A = real_dist(gen);
        stumps.push_back([&, i, A](Eigen::VectorXd x) {
            return (x(i) <= A) ? 1 : -1;
        });
    }
}

double AdaBoost::calcStumpAccuracy(int i) {
    double acc = 0;
    for (int j = 0; j < m; j++) {
        if (stumps[i](X.row(j)) == y(j))
            acc += w(j);
    }
    return acc;
}

int AdaBoost::classify(const Eigen::VectorXd &x) {
    double S = 0;
    for (int i = 0; i < n_stumps; i++)
        S += double(stumps[i](x)) * a(i);
    return (S >= 0) ? 1 : -1;
}

void AdaBoost::fit() {
    std::cout << "Preparing Stumps..." << std::endl;
    prepareStumps();

    for (int step = 0; step < n_steps; step++) {
        std::cout << "Step " << step+1 << "/" << n_steps << std::endl;

        int best_stump = 0;
        double best_acc = calcStumpAccuracy(0);

        for (int i = 0; i < n_stumps; i++) {
            double acc = calcStumpAccuracy(i);
            if (acc > best_acc) {
                best_acc = acc;
                best_stump = i;
            }
        }

        if (best_acc == 1)
            best_acc = 0.99;

        double alpha = log(best_acc / (1.0 - best_acc));
        a(best_stump) += alpha;

        for (int j = 0; j < m; j++)
            if (stumps[best_stump](X.row(j)) == y(j))
                w(j) *= exp(-alpha);
            else
                w(j) *= exp(alpha);

        w /= w.sum();
    }
}


