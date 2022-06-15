/* using phishing.data from miniproject 3 */

#include <bits/stdc++.h>
#include <Eigen/Dense>
#include "svm/svm.h"
#include "pca/pca.h"
#include "glcm/glcm.h"
#include "trees/tree.h"
#include "hog/hog.h"

using namespace std;

vector<vector<float>> useGlcmFeatures(const vector<vector<float>> &vt) {
    vector<vector<float>> newVt;
    for (const auto &vec : vt) {
        Glcm glcm = Glcm(3, 0, 15);
        const int sz = int(sqrt(vec.size()));
        vector<vector<double>> image(sz);
        for (int i = 1; i < vec.size(); ++i) {
            image[(i - 1) / sz].push_back(vec[i - 1]);
        }
        glcm.computeMatrix(image);
        auto hog_flat = glcm.getFlatMatrix();
        vector<float> floated;
        floated.push_back(vec.front());
        for (auto i : hog_flat)
            floated.push_back(i);
        //floated = make_compression(floated);
        newVt.push_back(floated);
    }
    return newVt;
}

vector<vector<float>> useHogFeatures(const vector<vector<float>> &vt) {
    vector<vector<float>> newVt;
    for (const auto &vec : vt) {
        Hog hog = Hog();
        const int sz = int(sqrt(vec.size()));
        vector<vector<double>> image(sz);
        for (int i = 1; i < vec.size(); ++i) {
            image[(i - 1) / sz].push_back(vec[i - 1]);
        }
        auto hog_flat = hog.getFlatHistogram(image);
        vector<float> floated;
        floated.push_back(vec.front());
        for (auto i : hog_flat)
            floated.push_back(i);
        //floated = make_compression(floated);
        newVt.push_back(floated);
    }
    return newVt;
}


int main() {
    int n_stumps = 10;
    int n_steps = 10000000;

    std::set<std::string> params;


    freopen("../../res/grey_scale_paintings.csv", "r", stdin);

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
        };
        //if (vect.back() == x1 || vect.back() == x2)
            v.push_back(vect);
    }
    std::cout << "readed\n";
    std::vector<std::vector<float> > A;
    for (auto row: v)
            A.push_back(row);
    std::shuffle(A.begin(), A.end(), std::mt19937(std::random_device()()));

    double train_frac = 0.92;
    double ver_frac = 0.0;

    std::vector<std::vector<float> > vt, vv, vtest;
    auto m1 = double(A.size());

    int a = train_frac * m1;
    int b = ver_frac * m1;
    for (int i = 0; i < a; i++)
        vt.push_back(A[i]);
    for (int i = a; i < a + b; i++)
        vv.push_back(A[i]);
    for (int i = a + b; i < m1; i++)
        vtest.push_back(A[i]);

    std::cout << "startGlcmFeatures\n";
    vt = useGlcmFeatures(vt);
    std::cout << "end glcm vt\n";
    vtest = useGlcmFeatures(vtest);
    std::cout << "end glcm vtest\n";

    /*//auto vt2 = useHogFeatures(vt);
    std::cout << "ended hog1\n";
    //auto vtest2 = useHogFeatures(vtest);
    std::cout << "ended hog2\n";
    for (auto i : vt2)
        vt.push_back(i);
    for (auto i : vtest2)
        vtest.push_back(i);
*/

    /* change observation labels from {0, 1} to {-1, 1} */
    //for (int i = 0; i < vt.size(); i++) vt[i][0] = vt[i][0] * 2 - 1;
    //for (int i = 0; i < vv.size(); i++) vv[i][0] = vv[i][0] * 2 - 1;
    //for (int i = 0; i < vtest.size(); i++) vtest[i][0] = vtest[i][0] * 2 - 1;

    /* copying labels into yt, yv, ytest vectors and deleting them from vectors */
    Eigen::VectorXd yt(vt.size());
    Eigen::VectorXd yv(vv.size());
    Eigen::VectorXd ytest(vtest.size());
    for (int i = 0; i < vt.size(); i++) {
        yt(i) = vt[i].front();
        std::cout << vt[i].front() << '\n';
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

    std::cout << "Training set: " << vt.size() << ", " << int(train_frac * m1) << "/" <<   "\n";
    std::cout << "Ver set:      " << vv.size() << ", " << int(ver_frac * m1) << "/" <<   "\n";
    std::cout << "Test set:     " << vtest.size() << ", " << m1 - int(train_frac * m1) - int(ver_frac * m1) << "\n";


    int d = vt[0].size();
    Eigen::MatrixXd Xt(vt.size(), d);
    Eigen::MatrixXd Xv(vv.size(), d);
    Eigen::MatrixXd Xtest(vtest.size(), d);

    for (int i = 0; i < vt.size(); i++)
        for (int j = 0; j < d; j++)
            Xt(i, j) = vt[i][j+1];

    for (int i = 0; i < vv.size(); i++)
        for (int j = 0; j < d; j++)
            Xv(i, j) = vv[i][j+1];

    for (int i = 0; i < vtest.size(); i++)
        for (int j = 0; j < d; j++)
            Xtest(i, j) = vtest[i][j+1];


    DecisionTree tree;
    tree.fit(Xt, yt);
    auto predicted = tree.predict(Xtest);
    double M = Xtest.rows();
    double correct = 0;
    std::cout<< "TESTING\n\n\n\n";
    for (int i = 0; i < M; i++) {
        int choisenClass = predicted[i];
        std::cout << choisenClass << " " << ytest(i) << '\n';
        if (choisenClass == int(ytest(i)))
            correct++;
    }
    std::cout << "accuracy on test set for tree: " << correct/M << "\n";
    exit(0);

    /*
    for (int i = 0; i < M; i++)
        if (adaBoost.classify(Xtest.row(i)) == ytest(i))
            correct++;
    std::cout << "accuracy on test data: " << correct/M << "\n";*/
}