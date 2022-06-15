#include <iostream>
#include <set>
#include <map>
#include <string>
#include <fstream>

#include <Eigen/Dense>

#include "svm/svm.h"
#include "hog/hog.h"
#include "tools.h"

const int N_CLASSES = 4;
const double TRAIN_FRACTION = 0.6;
const double VAL_FRACTION = 0.2;

int main(int argc, char *argv[]) {
    if (argc <= 1) {
        std::cout << "Usage: data_fraction [hog, glcm, pca, st, minmax]" << std::endl;
        return 0;
    }

    double data_fraction = std::stod(argv[1]);
    std::set<std::string> params;
    for (int i = 2; i < argc; i++)
        params.insert(argv[i]);

    std::cout << "Multiclass Linear SVM for " << data_fraction * 100 << "% of data" << std::endl;

    auto data = inputAndPrepareDataForMulticlassClassification(
                "../../res/grey_scale_paintings.csv", data_fraction, params, N_CLASSES, TRAIN_FRACTION, VAL_FRACTION, 0.75
    );

    Eigen::MatrixXd X_train = data.first[0];
    Eigen::MatrixXd X_val = data.first[1];
    Eigen::MatrixXd X_test = data.first[2];
    Eigen::VectorXd y_train = data.second[0];
    Eigen::VectorXd y_val = data.second[1];
    Eigen::VectorXd y_test = data.second[2];

    std::map<std::pair<int, int>, LinSVM*> svm;

    for (int i = 0; i < N_CLASSES; i++) {
        for (int j = i+1; j < N_CLASSES; j++) {
            /* svm for classes i and j */
            auto mats = extractDataForBinaryClassification(X_train, X_val, y_train, y_val, i, j);
            Eigen::MatrixXd _X_train = mats.first[0];
            Eigen::MatrixXd _X_val = mats.first[1];
            Eigen::VectorXd _y_train = mats.second[0];
            Eigen::VectorXd _y_val = mats.second[1];
            svm[{i, j}] = new LinSVM(_X_train.cols(), _X_train, _X_val, _y_train, _y_val);

            std::cout << "Teaching SVM for classes " << i << " and " << j << "..." << std::endl;

            svm[{i, j}]->fit();
        }
    }

    std::cout << "Classifying paintings from test set..." << std::endl;

    double goodGuesses = 0;
    for (int k = 0; k < X_test.rows(); k++) {
        Eigen::VectorXd x = X_test.row(k);
        std::vector<int> points(N_CLASSES, 0);
        for (int i = 0; i < N_CLASSES; i++) {
            for (int j = i+1; j < N_CLASSES; j++) {
                int pred = svm[{i, j}]->classify(x);
                if (pred == -1) points[i]++;
                else points[j]++;
            }
        }
        int predictedClass = distance(points.begin(), std::max_element(points.begin(), points.end()));
        if (predictedClass == y_test(k))
            goodGuesses++;
    }

    std::cout << "Test set accuracy for " << N_CLASSES << " classes: " << goodGuesses/double(X_test.rows()) << std::endl;
    std::cout << "Finished" << std::endl;
}