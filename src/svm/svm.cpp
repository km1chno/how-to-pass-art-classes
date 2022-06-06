#include <vector>
#include <algorithm>
#include <iostream>
#include <random>
#include <Eigen/Dense>

#include "svm.h"

const double EPS = 1e-6;
const int STEPS = 1000;

double LinSVM::calc_accuracy() {
    double correct = 0;
    for (int i = 0; i < m; i++) 
        if (yv(i)*classify(Xv.row(i)) > 0)
            correct++;
    return correct/m;
}

void LinSVM::SMO(double c) {
    a = Eigen::VectorXd::Zero(m);
    w = Eigen::VectorXd::Zero(d);
    b = 0;

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
            H = std::min(c/m, c/m - a(i) + a(j));
        } else {
            L = std::max(0.0, a(i) + a(j) - c/m);
            H = std::min(c/m, a(i) + a(j));
        }

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
    Eigen::MatrixXd _yt, Eigen::MatrixXd _yv
) : m(_Xt.rows()), d(_d), Xt(_Xt), Xv(_Xv), yt(_yt), yv(_yv) {}

int LinSVM::classify(const Eigen::VectorXd &x) {
    return (w.dot(x) >= 0) ? 1 : -1;
}

void LinSVM::fit(double low_c, double high_c, int n_c) {
    C = std::vector<double>(n_c);
    double c = low_c;
    generate(C.begin(), C.end(), [&c, low_c, high_c, n_c] { 
        double _c = c; c += (high_c-low_c)/double(n_c); return _c;
    });

    double best_acc = 0;
    double best_c = C[0];
    Eigen::VectorXd _w = Eigen::VectorXd::Zero(d);    
    double _b = 0;

    for (int i = 0; i < m; i++)
        for (int j = i; j < m; j++) 
            K(i, j) = K(j, i) = Xt.row(i).dot(Xt.row(j));

    for (auto c : C) {
        SMO(c);
        double acc = calc_accuracy();
        if (acc > best_acc) {
            best_c = c;
            best_acc = acc;
            _w = w;
            _b = b;
        }

        std::cout << "Accuracy for " << c << ": " << acc << "\n";
    }

    std::cout << "Best accuract for " << best_c << " was " << best_acc << "\n";

    w = _w;
    b = _b;
}

int main() {

}
        
