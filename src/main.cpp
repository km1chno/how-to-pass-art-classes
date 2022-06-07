/* using phishing.data from miniproject 3 */

#include <bits/stdc++.h>
#include <Eigen/Dense>
#include "svm.h"

using namespace std;

int main() {
    srand(time(NULL));

    std::string input_str;
    std::vector<std::vector<double> > v;

    while (std::getline( std::cin, input_str )) {
        std::stringstream ss(input_str);
        std::vector<double> vect;
        double i;

        while (ss >> i)
        {
            vect.push_back(i);

            if (ss.peek() == ',')
                ss.ignore();
        }

        v.push_back(vect);
    }

    std::vector<std::vector<double> > A, B;
    for (auto row : v)
        if (row.back() == 1)
            A.push_back(row);
        else 
            B.push_back(row);

    std::random_shuffle(A.begin(), A.end());
    std::random_shuffle(B.begin(), B.end());

    double train_frac = 0.6;
    double ver_frac = 0.2;

    std::vector<std::vector<double> > vt, vv, vtest;
    double m1 = double(A.size());
    double m2 = double(B.size());

    int a = train_frac * m1;
    int b = ver_frac * m1;
    for (int i = 0; i < a; i++) 
        vt.push_back(A[i]);
    for (int i = a; i < a+b; i++) 
        vv.push_back(A[i]);
    for (int i = a+b; i < m1; i++) 
        vtest.push_back(A[i]);
    
    a = train_frac * m2;
    b = ver_frac * m2;
    for (int i = 0; i < a; i++) 
        vt.push_back(B[i]);
    for (int i = a; i < a+b; i++) 
        vv.push_back(B[i]);
    for (int i = a+b; i < m2; i++) 
        vtest.push_back(B[i]);

    std::cout << "Training set: " << vt.size() << ", " << int(train_frac * m1) << "/" << int(train_frac * m2) << "\n";
    std::cout << "Ver set:      " << vv.size() << ", " << int(ver_frac * m1) << "/" << int(ver_frac * m2) << "\n";
    std::cout << "Test set:     " << vtest.size() << ", " << int(m1 - train_frac * m1 - ver_frac * m1) << "/" << int(m2 - train_frac * m2 - ver_frac * m2) << "\n";

    int d = vt[0].size()-1;
    Eigen::MatrixXd Xt(vt.size(), d);
    Eigen::MatrixXd Xv(vv.size(), d);
    Eigen::MatrixXd Xtest(vtest.size(), d);
    Eigen::VectorXd yt(vt.size());
    Eigen::VectorXd yv(vv.size());
    Eigen::VectorXd ytest(vtest.size());

    for (int i = 0; i < vt.size(); i++) {
        for (int j = 0; j < d-1; j++)
            Xt(i, j) = vt[i][j];
        yt(i) = vt[i].back();
    }
    for (int i = 0; i < vv.size(); i++) {
        for (int j = 0; j < d-1; j++)
            Xv(i, j) = vv[i][j];
        yv(i) = vv[i].back();
    }
    for (int i = 0; i < vtest.size(); i++) {
        for (int j = 0; j < d-1; j++)
            Xtest(i, j) = vtest[i][j];
        ytest(i) = vtest[i].back();
    }
    
    LinSVM svm(d, Xt, Xv, yt, yv);
    svm.fit(100.0, 300.0, 20);

    double M = Xtest.rows();
    double correct = 0;
    for (int i = 0; i < M; i++)
        if (svm.classify(Xtest.row(i)) == ytest(i))
            correct++;
    std::cout << "accuracy on test set: " << correct/M << "\n";
}