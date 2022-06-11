//#ifndef _IMAGES_HELPER_CPP_
//#define _IMAGES_HELPER_CPP_
#include <iostream>
#include <filesystem>
#define cimg_use_jpeg
#include "CImg.h"
#include "imagesHelper.h"

using namespace cimg_library;

typedef std::tuple<unsigned char,unsigned char,unsigned char> unsignedCharTuple;

CImg<unsigned char> resizeImage(const CImg<unsigned char> &image, int width, int height) {
    return std::move(image.get_resize(width, height));
}

std::string getResPath() {
    return std::string(std::filesystem::current_path()) + "/../../res/";
}

CImg<unsigned char> loadImageFromRes(const std::string& filename) {
    std::string fullPath = getResPath() + filename;
    return CImg<unsigned char>().get_load_jpeg(&fullPath[0]);
}

CImg<unsigned char> saveImageToRes(const std::string& filename, const CImg<unsigned char>& image) {
    std::string fullPath = getResPath() + filename;
    return image.save_jpeg(&fullPath[0]);
}

std::vector<unsignedCharTuple> convertImageToVectorRGB(const CImg<unsigned char>& image) {
    const int width = image.width();
    const int height = image.height();
    std::vector<unsignedCharTuple> res;
    for (int r = 0; r < height; ++r)
        for (int c = 0; c < width; ++c) {
            unsigned char reg = image(c, r, 0, 0),
                green = image(c, r, 0, 1),
                blue = image(c, r, 0, 2);
            res.emplace_back(reg, green, blue);
        }
    return res;
}

CImg<unsigned char> loadImageFromVectorRGB(const std::vector<unsignedCharTuple> &input, int width, int height) {
    CImg<unsigned char> result(width, height, 1, 3);
    int next = 0;
    for (int r = 0; r < height; ++r)
        for (int c = 0; c < width; ++c) {
            unsigned char red, green, blue;
            std::tie(red, green, blue) = input[next++];
            result(c, r, 0, 0) = red;
            result(c, r, 0, 1) = green;
            result(c, r, 0, 2) = blue;
        }
    return result;
}

void just_for_fun() {
    auto img = loadImageFromRes("exm.jpg");
    auto vec = convertImageToVectorRGB(img);
    auto imgnew = loadImageFromVectorRGB(vec, img.width(), img.height());
    saveImageToRes("exm2.jpg", imgnew);
    std::cout << "Hello, World!" << std::endl;
}

//#endif