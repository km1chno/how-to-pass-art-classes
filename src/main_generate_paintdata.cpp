#include <iostream>
#include "images/datasetManager.h"

int main(int argc, char *argv[]) {
    if (argc >= 2) {
        std::cout << "Preparing Dataset..." << std::endl;
        int n = std::stoi(argv[1]);
        deleteAugmentedDataset();
        prepareDataset(n, n, true);
    }

    std::cout << "Saving into /res/grey_scale_paintings.csv" << std::endl;
    saveDoubleRepresentedGenres();
}
