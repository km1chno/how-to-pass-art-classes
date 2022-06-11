#ifndef _CNN_
#define _CNN_

/* Based on https://github.com/mlpack/examples/blob/master/mnist_cnn/mnist_cnn.cpp */

#include <mlpack/core.hpp>
#include <mlpack/core/data/split_data.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <ensmallen.hpp>

using namespace mlpack;
using namespace mlpack::ann;
using namespace arma;
using namespace std;
using namespace ens;

class ConvolutionalNeuralNetwork {
    /* path to .csv file with data, required format:
     * rows - points
     * columns - the first column must be the label in range [0, classNum), the rest are the features
     * values in every row must be comma separated
     */
    std::string datasetPath;

    /* returns accuracy of model on data X with labels Y */
    double testModelOnData(FFN<NegativeLogLikelihood<>, RandomInitialization> model, mat X, mat Y);
    /* gets labels of predictions from model prediction object */
    arma::Row<size_t> getLabels(arma::mat predOut);

public:
    ConvolutionalNeuralNetwork(std::string _datasetPath);
    /* performs the whole algorithm of splitting data, learning and classifying
     * returns accuracy on test data
     */
    double fit();
};

#endif