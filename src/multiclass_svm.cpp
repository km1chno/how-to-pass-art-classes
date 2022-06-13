#include <iostream>
#include <set>
#include <map>
#include <string>
#include <fstream>
#include <random>

#include <Eigen/Dense>

#include "svm/svm.h"
#include "pca/pca.h"
#include "hog/hog.h"
#include "glcm/glcm.h"
#include "tools.h"

const int N_CLASSES = 4;
const double TRAIN_FRACTION = 0.6;
const double VAL_FRACTION = 0.2;

vector<vector<double>> useHogFeatures(const vector<vector<double>> &vt) {
    Hog hog = Hog();
    vector<vector<double>> newVt;
    for (const auto &vec : vt) {
        const int sz = int(sqrt(vec.size()));
        vector<vector<double>> image(sz);
        for (int i = 0; i + 1 < vec.size(); ++i) {
            image[i / sz].push_back(vec[i]);
        }
        auto hog_flat = hog.getFlatHistogram(image);
        newVt.push_back(hog_flat);
    }
    return newVt;
}

vector<vector<double>> useGlcmFeatures(const vector<vector<double>> &vt) {
    vector<vector<double>> newVt;
    for (const auto &vec : vt) {
        Glcm glcm = Glcm(3, 0, 15);
        const int sz = int(sqrt(vec.size()));
        vector<vector<double>> image(sz);
        for (int i = 0; i + 1 < vec.size(); ++i) {
            image[i / sz].push_back(vec[i]);
        }
        glcm.computeMatrix(image);
        auto glcm_flat = glcm.getAllFeaturesFromMatrix();
        newVt.push_back(glcm_flat);
    }
    return newVt;
}

std::vector<std::vector<double> > inputFromCSV(std::string path) {
    std::ifstream in(path);
    std::string input_str;
    std::vector<std::vector<double> > v;

    while (std::getline(in, input_str)) {
        std::stringstream ss(input_str);
        std::vector<double> vect;
        double i;
        while (ss >> i) {
            vect.push_back(i);
            if (ss.peek() == ',')
                ss.ignore();
        }
        v.push_back(vect);
    }
    return v;
}

void splitDataByClass(std::vector<std::vector<double> > &data, double fraction, std::vector<std::vector<double> > target[]) {
    for (auto &row : data)
        target[int(row.front())].push_back(std::vector<double>(row.begin()+1, row.end()));
    for (int i = 0; i < N_CLASSES; i++)
        target[i].erase(target[i].begin() + int(fraction * target[i].size()), target[i].end());
}

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

    std::vector<std::vector<double> > data = inputFromCSV("../res/grey_scale_paintings.csv");
    std::vector<std::vector<double> > splitData[N_CLASSES];
    splitDataByClass(data, data_fraction, splitData);

    /* random shuffle split data */
    for (int i = 0; i < N_CLASSES; i++)
        std::shuffle(splitData[i].begin(), splitData[i].end(), std::mt19937(std::random_device()()));

    std::vector<std::vector<double> > hog_features[N_CLASSES];
    std::vector<std::vector<double> > glcm_features[N_CLASSES];

    if (params.count("hog")) {
        std::cout << "Calculating HOG features...\n";
        for (int i = 0; i < N_CLASSES; i++)
            hog_features[i] = useHogFeatures(splitData[i]);
    }
    int n_glcm_features = 0;
    if (params.count("glcm")) {
        std::cout << "Calculating GLCM features...\n";
        for (int i = 0; i < N_CLASSES; i++)
            glcm_features[i] = useGlcmFeatures(splitData[i]);
        n_glcm_features = glcm_features[0][0].size();
    }

    for (int i = 0; i < N_CLASSES; i++) {
        for (int j = 0; j < hog_features[i].size(); j++)
            splitData[i][j].insert(splitData[i][j].end(), hog_features[i][j].begin(), hog_features[i][j].end());
        for (int j = 0; j < glcm_features[i].size(); j++)
            splitData[i][j].insert(splitData[i][j].end(), glcm_features[i][j].begin(), glcm_features[i][j].end());
    }

    std::vector<std::vector<double> > X_train, X_val, X_test;
    std::vector<double> y_train, y_val, y_test;
    for (int i = 0; i < N_CLASSES; i++) {
        for (int j = 0; j < TRAIN_FRACTION * splitData[i].size(); j++) {
            X_train.push_back(splitData[i][j]);
            y_train.push_back(i);
        }
        for (int j = TRAIN_FRACTION * splitData[i].size(); j < (TRAIN_FRACTION + VAL_FRACTION) * splitData[i].size(); j++) {
            X_val.push_back(splitData[i][j]);
            y_val.push_back(i);
        }
        for (int j = (TRAIN_FRACTION + VAL_FRACTION) * splitData[i].size(); j < splitData[i].size(); j++) {
            X_test.push_back(splitData[i][j]);
            y_test.push_back(i);
        }
    }

    std::cout << "Train set size =        " << X_train.size() << std::endl;
    std::cout << "Validation set size =   " << X_val.size() << std::endl;
    std::cout << "Test set size =         " << X_test.size() << std::endl;

    int d = X_train[0].size();

    Eigen::MatrixXd X_train_mat(X_train.size(), d);
    Eigen::VectorXd y_train_mat(X_train.size());
    Eigen::MatrixXd X_val_mat(X_val.size(), d);
    Eigen::VectorXd y_val_mat(y_val.size());
    Eigen::MatrixXd X_test_mat(X_test.size(), d);
    Eigen::VectorXd y_test_mat(y_test.size());

    for (int i = 0; i < X_train.size(); i++)
        X_train_mat.row(i) = Eigen::VectorXd::Map(&X_train[i][0], d);
    y_train_mat = Eigen::VectorXd::Map(&y_train[0], X_train.size());
    for (int i = 0; i < X_val.size(); i++)
        X_val_mat.row(i) = Eigen::VectorXd::Map(&X_val[i][0], d);
    y_val_mat = Eigen::VectorXd::Map(&y_val[0], X_val.size());
    for (int i = 0; i < X_test.size(); i++)
        X_test_mat.row(i) = Eigen::VectorXd::Map(&X_test[i][0], d);
    y_test_mat = Eigen::VectorXd::Map(&y_test[0], X_test.size());

    std::cout << "Number of features: " << X_train_mat.cols() << std::endl;

    if (params.count("st")) {
        std::cout << "Data Standarization..." << std::endl;
        standarizeData(X_train_mat, X_val_mat, X_test_mat);
    }
    if (params.count("minmax")) {
        std::cout << "Min-Max scaling..." << std::endl;
        standarizeData(X_train_mat, X_val_mat, X_test_mat);
    }
    if (params.count("pca")) {
        std::cout << "PCA for fraction of 65%..." << std::endl;

        /* we want to omit glcm features so its gonna take some engineering */
        Eigen::MatrixXd X_train_mat_noglcm = X_train_mat.block(0, 0, X_train_mat.rows(), X_train_mat.cols()-n_glcm_features);
        Eigen::MatrixXd X_val_mat_noglcm = X_val_mat.block(0, 0, X_val_mat.rows(), X_val_mat.cols()-n_glcm_features);
        Eigen::MatrixXd X_test_mat_noglcm = X_test_mat.block(0, 0, X_test_mat.rows(), X_test_mat.cols()-n_glcm_features);

        Eigen::MatrixXd feature_mat = pcaFraction(X_train_mat_noglcm, 0.65);

        int n_features = feature_mat.cols() + n_glcm_features;
        std::cout << "Number of features after PCA: " << n_features << std::endl;

        X_train_mat_noglcm *= feature_mat;
        X_val_mat_noglcm *= feature_mat;
        X_test_mat_noglcm *= feature_mat;

        /* extending _noglcm matrices to have space for glcm features */
        X_train_mat_noglcm.conservativeResize(X_train_mat_noglcm.rows(), X_train_mat_noglcm.cols() + n_glcm_features);
        X_val_mat_noglcm.conservativeResize(X_val_mat_noglcm.rows(), X_val_mat_noglcm.cols() + n_glcm_features);
        X_test_mat_noglcm.conservativeResize(X_test_mat_noglcm.rows(), X_test_mat_noglcm.cols() + n_glcm_features);

        /* copying from glcm features into _noglcm matrices */
        X_train_mat_noglcm.block(0, n_features-n_glcm_features, X_train_mat_noglcm.rows(), n_glcm_features)
            = X_train_mat.block(0, X_train_mat.cols()-n_glcm_features, X_train_mat.rows(), n_glcm_features);
        X_val_mat_noglcm.block(0, n_features-n_glcm_features, X_val_mat_noglcm.rows(), n_glcm_features)
            = X_val_mat.block(0, X_val_mat.cols()-n_glcm_features, X_val_mat.rows(), n_glcm_features);
        X_test_mat_noglcm.block(0, n_features-n_glcm_features, X_test_mat_noglcm.rows(), n_glcm_features)
            = X_test_mat.block(0, X_test_mat.cols()-n_glcm_features, X_test_mat.rows(), n_glcm_features);

        /* copying _noglcm matrices back into original */
        X_train_mat = X_train_mat_noglcm;
        X_val_mat = X_val_mat_noglcm;
        X_test_mat = X_test_mat_noglcm;

        /* standarization after PCA */
        standarizeData(X_train_mat, X_val_mat, X_test_mat);
    }

    d = X_train_mat.cols();

    std::map<std::pair<int, int>, LinSVM*> svm;

    for (int i = 0; i < N_CLASSES; i++) {
        for (int j = i+1; j < N_CLASSES; j++) {
            /* svm for classes i and j */
            int trainingSize = 0;
            int validationSize = 0;
            for (int k = 0; k < y_train_mat.size(); k++)
                if (y_train_mat(k) == i || y_train_mat(k) == j)
                    trainingSize++;
            for (int k = 0; k < y_val_mat.size(); k++)
                if (y_val_mat(k) == i || y_val_mat(k) == j)
                    validationSize++;

            Eigen::MatrixXd _X_train_mat(trainingSize, d);
            Eigen::VectorXd _y_train_mat(trainingSize);
            Eigen::MatrixXd _X_val_mat(validationSize, d);
            Eigen::VectorXd _y_val_mat(validationSize);

            int it = 0;
            for (int k = 0; k < X_train_mat.rows(); k++)
                if (y_train_mat(k) == i || y_train_mat(k) == j) {
                    _X_train_mat.row(it) = X_train_mat.row(k);
                    _y_train_mat(it++) = (y_train_mat(k) == i) ? -1 : 1;
                }
            it = 0;
            for (int k = 0; k < X_val_mat.rows(); k++)
                if (y_val_mat(k) == i || y_val_mat(k) == j) {
                    _X_val_mat.row(it) = X_val_mat.row(k);
                    _y_val_mat(it++) = (y_val_mat(k) == i) ? -1 : 1;
                }

            svm[{i, j}] = new LinSVM(d, _X_train_mat, _X_val_mat, _y_train_mat, _y_val_mat);
            std::cout << "Teaching SVM for classes " << i << " and " << j << "..." << std::endl;
            svm[{i, j}]->fit();
        }
    }

    std::cout << "Classifying paintings from test set..." << std::endl;

    double goodGuesses = 0;
    for (int k = 0; k < X_test_mat.rows(); k++) {
        Eigen::VectorXd x = X_test_mat.row(k);
        std::vector<int> points(N_CLASSES, 0);
        for (int i = 0; i < N_CLASSES; i++) {
            for (int j = i+1; j < N_CLASSES; j++) {
                int pred = svm[{i, j}]->classify(x);
                if (pred == -1) points[i]++;
                else points[j]++;
            }
        }
        int predictedClass = distance(points.begin(), std::max_element(points.begin(), points.end()));
        if (predictedClass == y_test_mat(k))
            goodGuesses++;
    }

    std::cout << "Test set accuracy for " << N_CLASSES << " classes: " << goodGuesses/double(X_test_mat.rows()) << std::endl;
    std::cout << "Finished" << std::endl;
}