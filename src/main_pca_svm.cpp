#include <bits/stdc++.h>
#include <Eigen/Dense>
#include <random>
#include "svm/svm.h"
#include "tools.h"
#include "pca/pca.h"
#include "hog/hog.h"

using namespace std;

/* vt are images without labels! */
vector<vector<float>> useHogFeatures(vector<vector<float>> vt) {
    Hog hog = Hog();
    vector<vector<float>> newVt;
    for (const auto &vec : vt) {
        const int sz = int(sqrt(vec.size()));
        vector<vector<float>> image(sz);
        for (int i = 0; i + 1 < vec.size(); ++i) {
            image[i / sz].push_back(vec[i]);
        }
        auto hog_flat = hog.getFlatHistogram(image);
        hog_flat.insert(hog_flat.end(), vec.begin(), vec.end());
        newVt.push_back(hog_flat);
    }
    vt = newVt;
    return newVt;
}

int main(int argc, char *argv[]) {
    /* {cubism 4000, impressionism 4000} 1 st pca 65% 2e7 steps, flip+rot - 65% acc */

    if (argc < 6) {
        std::cout << "Usage: class1 class2 min_c max_c data_fraction [hog, st, pca]\n";
        return 0;
    }
    int class1 = std::stoi(argv[1]);
    int class2 = std::stoi(argv[2]);
    float low = std::stof(argv[3]);
    float high = std::stof(argv[4]);
    float fraction = std::stof(argv[5]);

    std::set<std::string> params;
    for (int i = 4; i < argc; i++)
        params.insert(argv[i]);

    freopen("../res/grey_scale_paintings.csv", "r", stdin);

    std::string input_str;
    std::vector<std::vector<float> > v;

    while (std::getline(std::cin, input_str)) {
        std::stringstream ss(input_str);
        std::vector<float> vect;
        float i;
        while (ss >> i) {
            vect.push_back(i);
            if (ss.peek() == ',')
                ss.ignore();
        }
        v.push_back(vect);
    }

    std::vector<std::vector<float> > A, B;
    for (auto row: v)
        if (row.front() == class1)
            A.push_back(row);
        else if (row.front() == class2)
            B.push_back(row);
    std::shuffle(A.begin(), A.end(), std::mt19937(std::random_device()()));
    std::shuffle(B.begin(), B.end(), std::mt19937(std::random_device()()));

    double train_frac = 0.6;
    double ver_frac = 0.2;

    std::vector<std::vector<float> > vt, vv, vtest;
    auto m1 = double(A.size()) * fraction;
    auto m2 = double(B.size()) * fraction;

    int a = train_frac * m1;
    int b = ver_frac * m1;
    for (int i = 0; i < a; i++)
        vt.push_back(A[i]);
    for (int i = a; i < a + b; i++)
        vv.push_back(A[i]);
    for (int i = a + b; i < m1; i++)
        vtest.push_back(A[i]);

    a = train_frac * m2;
    b = ver_frac * m2;
    for (int i = 0; i < a; i++)
        vt.push_back(B[i]);
    for (int i = a; i < a + b; i++)
        vv.push_back(B[i]);
    for (int i = a + b; i < m2; i++)
        vtest.push_back(B[i]);

    /* change observation labels from {class1, class2} to {-1, 1} */
    for (int i = 0; i < vt.size(); i++) vt[i][0] = (vt[i][0] == class1) ? -1 : 1;
    for (int i = 0; i < vv.size(); i++) vv[i][0] = (vv[i][0] == class1) ? -1 : 1;
    for (int i = 0; i < vtest.size(); i++) vtest[i][0] = (vtest[i][0] == class1) ? -1 : 1;

    /* copying labels into yt, yv, ytest vectors and deleting them from vectors */
    Eigen::VectorXd yt(vt.size());
    Eigen::VectorXd yv(vv.size());
    Eigen::VectorXd ytest(vtest.size());
    for (int i = 0; i < vt.size(); i++) {
        yt(i) = vt[i].front();
        vt[i].erase(vt[i].begin());
    }
    for (int i = 0; i < vv.size(); i++) {
        yv(i) = vv[i].front();
        vv[i].erase(vv[i].begin());
    }
    for (int i = 0; i < vtest.size(); i++) {
        ytest(i) = vtest[i].front();
        vtest[i].erase(vtest[i].begin());
    }

    std::cout << "Training set: " << vt.size() << ", " << int(train_frac * m1) << "/" << int(train_frac * m2) << "\n";
    std::cout << "Ver set:      " << vv.size() << ", " << int(ver_frac * m1) << "/" << int(ver_frac * m2) << "\n";
    std::cout << "Test set:     " << vtest.size() << ", " << m1 - int(train_frac * m1) - int(ver_frac * m1) << "/" << m2 - int(train_frac * m2) - int(ver_frac * m2) << "\n";

    if (params.count("hog")) {
        std::cout << "Calculating HOG features...\n";
        vt = useHogFeatures(vt);
        vv = useHogFeatures(vv);
        vtest = useHogFeatures(vtest);
    }

    int d = vt[0].size();
    Eigen::MatrixXd Xt(vt.size(), d);
    Eigen::MatrixXd Xv(vv.size(), d);
    Eigen::MatrixXd Xtest(vtest.size(), d);

    for (int i = 0; i < vt.size(); i++)
        for (int j = 0; j < d; j++)
            Xt(i, j) = vt[i][j];

    for (int i = 0; i < vv.size(); i++)
        for (int j = 0; j < d; j++)
            Xv(i, j) = vv[i][j+1];

    for (int i = 0; i < vtest.size(); i++)
        for (int j = 0; j < d; j++)
            Xtest(i, j) = vtest[i][j+1];

    if (params.count("st")) {
        std::cout << "Data Standarization...\n";
        standarizeData(Xt, Xv, Xtest); // <- makes everything worse:(
    }
    if (params.count("minmax")) {
        std::cout << "Min-Max scaling...\n";
        minmaxscaling(Xt, Xv, Xtest);
    }
    if (params.count("pca")) {
        std::cout << "PCA for fraction of 65%\n";
        Eigen::MatrixXd feature_mat = pcaFraction(Xt, 0.65);
        std::cout << "number of features: " << feature_mat.cols() << " and samples " << feature_mat.rows() << "\n";
        Xt *= feature_mat;
        Xv *= feature_mat;
        Xtest *= feature_mat;
    }

    LinSVM svm(Xt.cols(), Xt, Xv, yt, yv);
    svm.fit(low, high, 5);

    double M = Xtest.rows();
    double correct = 0;
    for (int i = 0; i < M; i++)
        if (svm.classify(Xtest.row(i)) == ytest(i))
            correct++;
    std::cout << "accuracy on test data: " << correct/M << "\n";
}