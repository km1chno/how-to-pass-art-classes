#include <iostream>
#include <fstream>
#include <filesystem>
#include "imagesHelper.h"
#include <CImg-3.1.3/CImg.h>

#define cimg_use_jpeg

using namespace cimg_library;

const int N_GENRES = 4;

const std::vector<int> numberOfPaintings = {
        3000, 3000, 3000, 3000
};

const std::vector<std::string> paintingGenres = {
        "Cubism", "Post-Impressionism", "Northern-Renaissance", "Pop-Art"
};

void saveDoubleRepresentedGenres() {
    std::string outputFileName = "../../res/grey_scale_paintings.csv";
    std::ofstream out(outputFileName);

    for (int i = 0; i < N_GENRES; i++) {
        const std::string& genre = paintingGenres[i];
        int n_arts = numberOfPaintings[i];

        int paintingsWritten = 0;
        for (const auto & entry : std::filesystem::directory_iterator("../../res/GreyScaleDataset/" + genre + "/")) {
            auto dir = std::string(entry.path());
            dir = dir.substr(3, dir.size() - 1);
            auto img = loadImageFromRes(dir);
            out << i;
            for (int j = 0; j < img.width(); j++)
                for (int k = 0; k < img.height(); k++)
                    out << "," << int(img(j, k, 0));
            out << "\n";
            if ((++paintingsWritten) == n_arts)
                break;
        }
    }
    out.close();
}

void deleteAugmentedDataset() {
    for (const auto& genre : paintingGenres) {
        for (const auto & entry : std::filesystem::directory_iterator("../../res/AugmentedDataset/" + genre + "/"))
            std::filesystem::remove_all(entry);
        for (const auto & entry : std::filesystem::directory_iterator("../../res/GreyScaleDataset/" + genre + "/"))
            std::filesystem::remove_all(entry);
    }
}

void prepareDataset(int width, int height, bool augment) {
    for (const auto& genre : paintingGenres) {
        /* Dataset -> AugmentedDataset */
        std::cout << "Dataset -> AugmentedDataset" << std::endl;
        for (const auto & entry : std::filesystem::directory_iterator("../../res/Dataset/" + genre + "/")) {
            std::string path = entry.path();
            path = path.substr(18, path.size()-22);
            std::cout << path << std::endl;
            copyImageIntoAugmentedDataset(path, width, height);
            if (augment) {
                generateFlippedImages(path, width, height);
                generateRotatedImages(path, width, height);
                generateCroppedImages(path, width, height, 3);
                generateSaltAndPepperNoiseImage(path, width, height, 0.1);
            }
        }
        /* AugmentedDataset -> GreyScaleDataset */
        std::cout << "AugmentedDataset -> GreyScaleDataset" << std::endl;
        for (const auto & entry : std::filesystem::directory_iterator("../../res/AugmentedDataset/" + genre + "/")) {
            std::string path = entry.path();
            path = path.substr(27, path.size()-31);
            std::cout << path << std::endl;
            generateGrayScaleImage(path);
        }
    }
}
