/* Based on https://github.com/mlpack/examples/blob/master/mnist_cnn/mnist_cnn.cpp */

#include <mlpack/core.hpp>
#include <mlpack/core/data/split_data.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <ensmallen.hpp>
#include "cnn.h"

using namespace mlpack;
using namespace mlpack::ann;
using namespace arma;
using namespace std;
using namespace ens;

ConvolutionalNeuralNetwork::ConvolutionalNeuralNetwork(std::string _datasetPath) : datasetPath(_datasetPath) {}

arma::Row<size_t> ConvolutionalNeuralNetwork::getLabels(arma::mat predOut) {
    arma::Row<size_t> predLabels(predOut.n_cols);
    for (arma::uword i = 0; i < predOut.n_cols; ++i)
        predLabels(i) = predOut.col(i).index_max() + 1;
    return predLabels;
}

double ConvolutionalNeuralNetwork::testModelOnData(FFN<NegativeLogLikelihood<>, RandomInitialization> model, mat X, mat Y) {
    mat testPredOut;
    model.Predict(X, testPredOut);
    Row<size_t> testPred = getLabels(testPredOut);
    return arma::accu(testPred == Y) / (double) Y.n_elem;
}

double ConvolutionalNeuralNetwork::fit() {
    srand(time(NULL));
    mlpack::math::RandomSeed(rand());

    constexpr int MAX_ITERATIONS = 0;
    constexpr double STEP_SIZE = 1.2e-3;
    constexpr int BATCH_SIZE = 64;

    mat dataset;
    data::Load(datasetPath, dataset, true);

    mat train, valid, test, temp;
    data::Split(dataset, train, temp, 0.4);
    data::Split(temp, valid, test, 0.5);

    const mat trainX = train.submat(1, 0, train.n_rows - 1, train.n_cols - 1);
    const mat validX = valid.submat(1, 0, valid.n_rows - 1, valid.n_cols - 1);
    const mat testX = test.submat(1, 0, test.n_rows - 1, test.n_cols - 1);

    const mat trainY = train.row(0) + 1;
    const mat validY = valid.row(0) + 1;
    const mat testY = test.row(0) + 1;

    size_t numIterations = trainX.n_cols * MAX_ITERATIONS;

    FFN<NegativeLogLikelihood<>, RandomInitialization> model;

    model.Add<Convolution<>>(1, 6, 9, 9, 1, 1, 0, 0, 52, 52);
    model.Add<LeakyReLU<>>();
    model.Add<MaxPooling<>>(2, 2, 2, 2, true);
    model.Add<Convolution<>>(6, 16, 5, 5, 1, 1, 0, 0, 22, 22);
    model.Add<LeakyReLU<>>();
    model.Add<MaxPooling<>>(2, 2, 2, 2, true);
    model.Add<Linear<>>(16 * 9 * 9, 2);
    model.Add<LogSoftMax<>>();

    cout << "Start training ..." << endl;

    // Set parameters for the Adam optimizer.
    ens::Adam optimizer(
      STEP_SIZE,  // Step size of the optimizer.
      BATCH_SIZE, // Batch size. Number of data points that are used in each
                  // iteration.
      0.9,        // Exponential decay rate for the first moment estimates.
      0.999, // Exponential decay rate for the weighted infinity norm estimates.
      1e-8,  // Value used to initialise the mean squared gradient parameter.
      numIterations, // Max number of iterations.
      1e-8,           // Tolerance.
      true);

    model.Train(trainX,
              trainY,
              optimizer,
              ens::PrintLoss(),
              ens::ProgressBar(),
              ens::EarlyStopAtMinLoss(
                  [&](const arma::mat&)
                  {
                    double validationLoss = model.Evaluate(validX, validY);
                    std::cout << "Validation loss: " << validationLoss
                        << "." << std::endl;
                    return validationLoss;
                  }));

    double testAcc = testModelOnData(model, testX, testY);
    std::cout << "Accuracy: train = " << testModelOnData(model, trainX, trainY) * 100 << "%" << std::endl;
    std::cout << "Accuracy: validation = " << testModelOnData(model, validX, validY) * 100 << "%" << std::endl;
    std::cout << "Accuracy: test = " << testAcc * 100 << "%" << std::endl;

    return testAcc;
}