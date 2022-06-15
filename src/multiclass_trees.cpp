#include <iostream>
#include <set>
#include <map>
#include <string>
#include <fstream>

#include <Eigen/Dense>

#include "glcm/glcm.h"
#include "trees/tree.h"
#include "hog/hog.h"
#include "tools.h"

const int N_CLASSES = 4;
const double TRAIN_FRACTION = 0.4;
const double VAL_FRACTION = 0.3;

int main(int argc, char *argv[]) {
    double data_fraction = 0.6;
    std::set<std::string> params;
    //params.insert("hog");
    params.insert("glcm");
    //params.insert("pca");

    std::cout << "Multiclass Linear SVM for " << data_fraction * 100 << "% of data" << std::endl;

    auto data = inputAndPrepareDataForMulticlassClassification(
            "../../res/grey_scale_paintings.csv", data_fraction, params, N_CLASSES, TRAIN_FRACTION, VAL_FRACTION, 0.65
    );

    Eigen::MatrixXd X_train = data.first[0];
    Eigen::MatrixXd X_val = data.first[1];
    Eigen::MatrixXd X_test = data.first[2];
    Eigen::VectorXd y_train = data.second[0];
    Eigen::VectorXd y_val = data.second[1];
    Eigen::VectorXd y_test = data.second[2];

    std::map<std::pair<int, int>, DecisionTree*> ab;
    std::map<std::pair<int, int>, double> error;
    for (int i = 0; i < N_CLASSES; i++) {
        for (int j = i+1; j < N_CLASSES; j++) {
            /* DecisionTrees for classes i and j */
            auto mats = extractDataForBinaryClassification(X_train, X_val, y_train, y_val, i, j);
            Eigen::MatrixXd _X_train = mats.first[0];
            Eigen::MatrixXd _X_val = mats.first[1];
            Eigen::VectorXd _y_train = mats.second[0];
            Eigen::VectorXd _y_val = mats.second[1];

                ab[{i, j}] = new DecisionTree();

            std::cout << "Teaching DecisionTrees for classes " << i << " and " << j << "..." << std::endl;

            ab[{i, j}]->fit(_X_train, _y_train);
            /* verification accuracy */
            int goodGuesses = 0;
            for (int k = 0; k < _X_val.rows(); k++) {
                //std::cout << "TEST " << ab[{i, j}]->predict(_X_val.row(k)) << " " << _y_val(k) << '\n';
                int predicted = ab[{i, j}]->predict(_X_val.row(k));
                //std::cout << predicted << " " << i << " " << j << " " << _y_val(k) << " : ";
                //std::cout << predicted << "\n";
                if  (predicted == _y_val(k))
                    goodGuesses++;
            }
            double local_error = goodGuesses/double(_X_val.rows());
            error[{i, j}] = local_error;
            std::cout << "Validation accuracy for classes (" << i << ", " << j << ") = " << local_error << std::endl;
        }
    }

    std::cout << "Classifying paintings from test set..." << std::endl;
    auto dec = DecisionTree();
    dec.fit(X_train, y_train);
    double goodGuesses = 0;
    int good2 = 0;
    int smartGoodGuesses = 0;
    for (int k = 0; k < X_test.rows(); k++) {
        Eigen::VectorXd x = X_test.row(k);
        std::vector<int> points(N_CLASSES, 0);
        std::vector<double> smart_points(N_CLASSES, 0);

        for (int i = 0; i < N_CLASSES; i++) {
            for (int j = i+1; j < N_CLASSES; j++) {
                int pred = ab[{i, j}]->predict(x);
                if (pred == -1) {
                    points[i]++;
                    smart_points[i] += error[{i, j}];
                    smart_points[j] += 1 - error[{i, j}];
                }
                else {
                    points[j]++;
                    smart_points[j] += error[{i, j}];
                    smart_points[i] += 1 - error[{i, j}];
                }

            }
        }

        int predictedClass = distance(points.begin(), std::max_element(points.begin(), points.end()));
        int smartPredictedClass = distance(smart_points.begin(), std::max_element(smart_points.begin(), smart_points.end()));
        int second = -1;
        for (int i = 0; i < points.size(); ++i)
            if (points[i] == points[predictedClass] && i != predictedClass) {
                second = i;
                break;
            }
        if (second != -1) {
            predictedClass = (ab[{predictedClass, second}]->predict(x) == -1 ? predictedClass : second);
            // better than just random
        }
        int val = dec.predict(x);
        if (predictedClass == y_test(k))
            goodGuesses++;
        if (val == y_test(k))
            ++good2;
        if (smartPredictedClass == y_test(k))
            ++smartGoodGuesses;
    }

    std::cout << "Test set accuracy for " << N_CLASSES << " classes: " << goodGuesses/double(X_test.rows()) << "| (smart) "<<smartGoodGuesses/double(X_test.rows()) << " " <<  "and in the main treee: " << good2 / double(X_test.rows()) << std::endl;
    std::cout << "Finished" << std::endl;
}