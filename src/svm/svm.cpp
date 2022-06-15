#include <vector>
#include <algorithm>
#include <iostream>
#include <random>
#include "Dense"

#include "svm.h"

const double EPS = 1e-6;
const int STEPS = 1000000;

double LinSVM::calc_accuracy() {
    double correct = 0;
    for (int i = 0; i < Xv.rows(); i++) 
        if (yv(i)*classify(Xv.row(i)) > 0)
            correct++;
    return correct/Xv.rows();
}

void LinSVM::SMO(double c) {
    a = Eigen::VectorXd::Zero(m);
    w = Eigen::VectorXd::Zero(d);
    b = 0;
    double _m = m;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, m-1);
    for (int step = 0; step < STEPS; step++) {
        int i = distrib(gen);
        int j = distrib(gen);
        while (i == j)
            j = distrib(gen);
        
        double L, H; 
        L = H = 0;
        if (yt(i) != yt(j)) {
            L = std::max(0.0, a(j) - a(i));
            H = std::min(c/_m, c/_m - a(i) + a(j));
        } else {
            L = std::max(0.0, a(i) + a(j) - c/_m);
            H = std::min(c/_m, a(i) + a(j));
        }

        K(i, j) = K(j, i) = Xt.row(i).dot(Xt.row(j));
        K(i, i) = Xt.row(i).dot(Xt.row(i));
        K(j, j) = Xt.row(j).dot(Xt.row(j));
        double A = 2*K(i, j) - K(i, i) - K(j, j);
        if (!A)
            continue;

        double E_i = w.dot(Xt.row(i)) - yt(i);
        double E_j = w.dot(Xt.row(j)) - yt(j);
        
        double old_j = a(j);
        double old_i = a(i);
        a(j) += yt(j)*(E_j - E_i)/A;
        a(j) = std::min(H, std::max(a(j), L));
        a(i) += yt(i)*yt(j)*(old_j - a(j));

        w += yt(i)*(a(i)-old_i)*Xt.row(i);
        w += yt[j]*(a[j]-old_j)*Xt.row(j);
    }
    b = yt(0) - w.dot(Xt.row(0));
    for (int i = 0; i < m; i++) 
        if (a(i) > EPS and a(i) + EPS < c/m) {
            b = yt(i) - w.dot(Xt.row(i));
            break;
        }
}

LinSVM::LinSVM(
    int _d,
    Eigen::MatrixXd _Xt, Eigen::MatrixXd _Xv,
    Eigen::VectorXd _yt, Eigen::VectorXd _yv
) : m(_Xt.rows()), d(_d), Xt(_Xt), Xv(_Xv), yt(_yt), yv(_yv) {
    K = Eigen::MatrixXd::Zero(m, m);
}

int LinSVM::classify(const Eigen::VectorXd &x) {
    return (w.dot(x)+b > 0) ? 1 : -1;
}

void LinSVM::fit() {
    C = std::vector<double>({0.1, 1, 2, 5, 10, 100, 500, 1000, 5000, 10000});

    double best_acc = 0;
    double best_c = C[0];
    Eigen::VectorXd _w = Eigen::VectorXd::Zero(d);    
    double _b = 0;

    for (auto c : C) {
        SMO(c);
        double acc = calc_accuracy();
        //std::cout << acc << '\n';
        if (acc > best_acc) {
            best_c = c;
            best_acc = acc;
            _w = w;
            _b = b;
        } else break;

        std::cout << "Validation accuracy for c = " << c << ": " << acc << "\n";
    }

    std::cout << "Best validation accuracy for c = " << best_c << " was " << best_acc << "\n";

    w = _w;
    b = _b;
}