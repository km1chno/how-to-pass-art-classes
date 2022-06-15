#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <set>
#include <random>

#include "tools.h"
#include "hog/hog.h"
#include "glcm/glcm.h"
#include "pca/pca.h"

void standarizeData(Eigen::MatrixXd &A, Eigen::MatrixXd &B, Eigen::MatrixXd &C) {
    Eigen::VectorXd mean = A.colwise().mean();
    Eigen::VectorXd stdev = Eigen::VectorXd::Zero(A.cols());
    for (int i = 0; i < A.rows(); i++)
        for (int j = 0; j < A.cols(); j++) {
            A(i, j) -= mean(j);
            stdev(j) += A(i, j)*A(i, j);
        }
    for (int j = 0; j < A.cols(); j++)
        stdev(j) = std::sqrt(stdev(j)/A.rows());

    for (int i = 0; i < A.rows(); i++)
        for (int j = 0; j < A.cols(); j++)
            A(i, j) /= stdev(j);

    for (int i = 0; i < B.rows(); i++)
        for (int j = 0; j < B.cols(); j++)
            B(i, j) = (B(i, j) - mean(j))/stdev(j);

    for (int i = 0; i < C.rows(); i++)
        for (int j = 0; j < C.cols(); j++)
            C(i, j) = (C(i, j) - mean(j))/stdev(j);
}

void minmaxscaling(Eigen::MatrixXd &A, Eigen::MatrixXd &B, Eigen::MatrixXd &C) {
    Eigen::VectorXd maxVal = A.row(0);
    Eigen::VectorXd minVal = A.row(0);

    for (int i = 1; i < A.rows(); i++)
        for (int j = 0; j < A.cols(); j++) {
            maxVal(j) = std::max(maxVal(j), A(i, j));
            minVal(j) = std::min(minVal(j), A(i, j));
        }

    for (int i = 0; i < A.rows(); i++)
        for (int j = 0; j < A.cols(); j++)
            A(i, j) = (A(i, j) - minVal(j)) / (maxVal(j) - minVal(j));

    for (int i = 0; i < B.rows(); i++)
        for (int j = 0; j < B.cols(); j++)
            B(i, j) = (B(i, j) - minVal(j)) / (maxVal(j) - minVal(j));

    for (int i = 0; i < C.rows(); i++)
        for (int j = 0; j < C.cols(); j++)
            C(i, j) = (C(i, j) - minVal(j)) / (maxVal(j) - minVal(j));
}

std::vector<std::vector<double> > inputFromCSV(const std::string &path) {
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

void splitDataByClass(std::vector<std::vector<double> > &data, double fraction, std::vector<std::vector<double> > target[], const int N_CLASSES) {
    for (auto &row : data)
        target[int(row.front())].emplace_back(row.begin()+1, row.end());
    for (int i = 0; i < N_CLASSES; i++)
        target[i].erase(target[i].begin() + int(fraction * double(target[i].size())), target[i].end());
}

std::pair<std::vector<Eigen::MatrixXd>, std::vector<Eigen::VectorXd> > inputAndPrepareDataForMulticlassClassification(
        const std::string &file_path,
        const double data_fraction,
        std::set<std::string> &params,
        const int N_CLASSES,
        const double TRAIN_FRACTION,
        const double VAL_FRACTION,
        const double PCA_FRACTION = 0.65
) {
    std::vector<std::vector<double> > data = inputFromCSV(file_path);
    std::vector<std::vector<double> > splitData[N_CLASSES];
    splitDataByClass(data, data_fraction, splitData, N_CLASSES);

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
        //splitData[i].clear();
        for (int j = 0; j < hog_features[i].size(); j++) {
            if (splitData[i].size() == j)
                splitData[i].emplace_back();
            splitData[i][j].insert(splitData[i][j].end(), hog_features[i][j].begin(), hog_features[i][j].end());
        }
        for (int j = 0; j < glcm_features[i].size(); j++) {
            if (splitData[i].size() == j)
                splitData[i].emplace_back();
            splitData[i][j].insert(splitData[i][j].end(), glcm_features[i][j].begin(), glcm_features[i][j].end());
        }
    }

    std::vector<std::vector<double> > X_train, X_val, X_test;
    std::vector<double> y_train, y_val, y_test;
    for (int i = 0; i < N_CLASSES; i++) {
        double size = splitData[i].size();
        for (int j = 0; j < TRAIN_FRACTION * size; j++) {
            X_train.push_back(splitData[i][j]);
            y_train.push_back(i);
        }
        for (int j = TRAIN_FRACTION * size; j < (TRAIN_FRACTION + VAL_FRACTION) * size; j++) {
            X_val.push_back(splitData[i][j]);
            y_val.push_back(i);
        }
        for (int j = (TRAIN_FRACTION + VAL_FRACTION) * size; j < size; j++) {
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

    if (params.count("pca")) {
        /* standarization before PCA */
        standarizeData(X_train_mat, X_val_mat, X_test_mat);

        std::cout << "PCA for fraction of " << int(PCA_FRACTION * 100) << "%..." << std::endl;

        /* we want to omit glcm features so its gonna take some engineering */
        Eigen::MatrixXd X_train_mat_noglcm = X_train_mat.block(0, 0, X_train_mat.rows(), X_train_mat.cols()-n_glcm_features);
        Eigen::MatrixXd X_val_mat_noglcm = X_val_mat.block(0, 0, X_val_mat.rows(), X_val_mat.cols()-n_glcm_features);
        Eigen::MatrixXd X_test_mat_noglcm = X_test_mat.block(0, 0, X_test_mat.rows(), X_test_mat.cols()-n_glcm_features);

        Eigen::MatrixXd feature_mat = pcaFraction(X_train_mat_noglcm, PCA_FRACTION);

        int n_features = int(feature_mat.cols()) + n_glcm_features;
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
    if (params.count("st")) {
        std::cout << "Data Standarization..." << std::endl;
        standarizeData(X_train_mat, X_val_mat, X_test_mat);
    }
    if (params.count("minmax")) {
        std::cout << "Min-Max scaling..." << std::endl;
        standarizeData(X_train_mat, X_val_mat, X_test_mat);
    }

    return {std::vector<Eigen::MatrixXd>({X_train_mat, X_val_mat, X_test_mat}), std::vector<Eigen::VectorXd>({y_train_mat, y_val_mat, y_test_mat})};
}

std::pair<std::vector<Eigen::MatrixXd>, std::vector<Eigen::VectorXd> > extractDataForBinaryClassification(
       Eigen::MatrixXd &X_train,
       Eigen::MatrixXd &X_val,
       Eigen::VectorXd &y_train,
       Eigen::VectorXd &y_val,
       int i,
       int j
) {
    int trainingSize = 0;
    int validationSize = 0;
    for (int k = 0; k < y_train.size(); k++)
        if (y_train(k) == i || y_train(k) == j)
            trainingSize++;
    for (int k = 0; k < y_val.size(); k++)
        if (y_val(k) == i || y_val(k) == j)
            validationSize++;

    Eigen::MatrixXd _X_train_mat(trainingSize, X_train.cols());
    Eigen::VectorXd _y_train_mat(trainingSize);
    Eigen::MatrixXd _X_val_mat(validationSize, X_train.cols());
    Eigen::VectorXd _y_val_mat(validationSize);

    int it = 0;
    for (int k = 0; k < X_train.rows(); k++)
        if (y_train(k) == i || y_train(k) == j) {
            _X_train_mat.row(it) = X_train.row(k);
            _y_train_mat(it++) = (y_train(k) == i) ? -1 : 1;
        }
    it = 0;
    for (int k = 0; k < X_val.rows(); k++)
        if (y_val(k) == i || y_val(k) == j) {
            _X_val_mat.row(it) = X_val.row(k);
            _y_val_mat(it++) = (y_val(k) == i) ? -1 : 1;
        }

    return {std::vector<Eigen::MatrixXd>({_X_train_mat, _X_val_mat}), std::vector<Eigen::VectorXd>({_y_train_mat, _y_val_mat})};
}