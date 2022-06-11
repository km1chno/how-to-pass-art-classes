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
    constexpr int BATCH_SIZE = 50;

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

    FFN<NegativeLogLikelihood<>, RandomInitialization> model;

    // Layers schema.
    // 28x28x1 --- conv (6 filters of size 5x5. stride = 1) ---> 24x24x6
    // 24x24x6 --------------- Leaky ReLU ---------------------> 24x24x6
    // 24x24x6 --- max pooling (over 2x2 fields. stride = 2) --> 12x12x6
    // 12x12x6 --- conv (16 filters of size 5x5. stride = 1) --> 8x8x16
    // 8x8x16  --------------- Leaky ReLU ---------------------> 8x8x16
    // 8x8x16  --- max pooling (over 2x2 fields. stride = 2) --> 4x4x16
    // 4x4x16  ------------------- Dense ----------------------> 10

    // Add the first convolution layer.
    model.Add<Convolution<>>(1,  // Number of input activation maps.
                           6,  // Number of output activation maps.
                           5,  // Filter width.
                           5,  // Filter height.
                           1,  // Stride along width.
                           1,  // Stride along height.
                           0,  // Padding width.
                           0,  // Padding height.
                           28, // Input width.
                           28  // Input height.
    );

    // Add first ReLU.
    model.Add<LeakyReLU<>>();

    // Add first pooling layer. Pools over 2x2 fields in the input.
    model.Add<MaxPooling<>>(2, // Width of field.
                          2, // Height of field.
                          2, // Stride along width.
                          2, // Stride along height.
                          true);

    // Add the second convolution layer.
    model.Add<Convolution<>>(6,  // Number of input activation maps.
                           16, // Number of output activation maps.
                           5,  // Filter width.
                           5,  // Filter height.
                           1,  // Stride along width.
                           1,  // Stride along height.
                           0,  // Padding width.
                           0,  // Padding height.
                           12, // Input width.
                           12  // Input height.
    );

    // Add the second ReLU.
    model.Add<LeakyReLU<>>();

    // Add the second pooling layer.
    model.Add<MaxPooling<>>(2, 2, 2, 2, true);

    // Add the final dense layer.
    model.Add<Linear<>>(16 * 4 * 4, 10);
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
      MAX_ITERATIONS, // Max number of iterations.
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