#include "cnn.h"

int main() {
    ConvolutionalNeuralNetwork cnn("../../res/grey_scale_paintings.csv");
    cnn.fit();
}