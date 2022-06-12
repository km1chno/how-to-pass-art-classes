#include <iostream>
#include <fstream>
#include <filesystem>
#include "imagesHelper.h"

using namespace cimg_library;

const std::tuple<std::string, std::string, int> genres[] =
        {{"Post-Impressionism", "Vincent_van_Gogh_", 400},
         {"Cubism", "Pablo_Picasso_", 400}};

const std::vector<std::string> paintingGenres = {
        "Cubism", "Post-Impressionism"
};

void resizeAllImagesFromRes(int width = 100, int height = 100) {
    std::cerr << "WIDTH HEIGHT: " << width << " " << height << '\n';
    for (auto item : genres) {
        std::string gener, author;
        int n_arts;
        std::tie(gener, author, n_arts) = item;
        std::string path = gener + "/" + author;
        for (int i = 1; i <= n_arts; ++i) {
            std::string filenameOld = path + std::to_string(i) + ".jpg";
            std::string filenameNew = path + std::to_string(i) + "NEW.jpg";
            auto img = loadImageFromRes(filenameOld);
            saveImageToRes(filenameNew, resizeImage(img, width, height));
        }
        //auto resizedImage = resizeImage(loadImageFromRes());
    }
}

void saveDoubleRepresentedSingleImage(const std::vector<unsignedCharTuple>& imageVector, std::ofstream &out) {
    for (auto item : imageVector) {
        unsigned char red, green, blue;
        std::tie(red, green, blue) = item;
        out << short(red) << " " << short(green) <<  " " << short(blue) << " ";
    }
    out << '\n';
}

void saveDoubleRepresentedGenres() {
    for (auto item : genres) {
        std::string gener, author;
        int n_arts;
        std::tie(gener, author, n_arts) = item;
        std::string path = gener + "/_";
        std::string outputFileName = getResPath() + path + "Representation";
        std::ofstream out(outputFileName);
        out << n_arts << '\n';

        for (int i = 1; i <= n_arts; ++i) {
            std::string path = gener + "/" + author + std::to_string(i);
            std::string filenameNew = path + "NEW.jpg";
            const auto &&img = loadImageFromRes(filenameNew);
            if (i == 1) {
                // we assume all images are prepared with the same size
                out << img.width() << ' ' << img.height() << '\n';
            }
            saveDoubleRepresentedSingleImage(convertImageToVectorRGB(img), out);
        }
        out.close();
    }
}

void deleteAugmentedDataset() {
    for (const auto& genre : paintingGenres) {
        for (const auto & entry : std::filesystem::directory_iterator("../res/AugmentedDataset/" + genre + "/"))
            std::filesystem::remove_all(entry);
        for (const auto & entry : std::filesystem::directory_iterator("../res/GreyScaleDataset/" + genre + "/"))
            std::filesystem::remove_all(entry);
    }
}

void prepareDataset(int width, int height, bool augment) {
    for (const auto& genre : paintingGenres) {
        /* Dataset -> AugmentedDataset */
        std::cout << "Dataset -> AugmentedDataset" << std::endl;
        for (const auto & entry : std::filesystem::directory_iterator("../res/Dataset/" + genre + "/")) {
            std::string path = entry.path();
            path = path.substr(15, path.size()-19);
            std::cout << path << std::endl;
            copyImageIntoAugmentedDataset(path, 52, 52);
            if (augment) {
                generateFlippedImages(path, 52, 52);
                generateRotatedImages(path, 52, 52);
                generateCroppedImages(path, 52, 52, 3);
            }
        }
        /* AugmentedDataset -> GreyScaleDataset */
        std::cout << "AugmentedDataset -> GreyScaleDataset" << std::endl;
        for (const auto & entry : std::filesystem::directory_iterator("../res/AugmentedDataset/" + genre + "/")) {
            std::string path = entry.path();
            path = path.substr(24, path.size()-28);
            std::cout << path << std::endl;
            generateGrayScaleImage(path);
        }
    }
}
