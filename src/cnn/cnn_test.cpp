#include "cnn.h"

int main() {
    ConvolutionalNeuralNetwork cnn("../../res/paintings.csv");
    cnn.fit();
}