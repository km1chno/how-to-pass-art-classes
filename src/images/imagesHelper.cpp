#ifndef _IMAGES_HELPER_HEADER_
#define _IMAGES_HELPER_HEADER_
#include <iostream>
#include <filesystem>
#define cimg_use_jpeg
#include "CImg.h"
using namespace cimg_library;


static CImg<unsigned char> resizeImage(const CImg<unsigned char> &image, int width, int height) {
    return std::move(image.get_resize(width, height));
}

static CImg<unsigned char> loadImageFromRes(const std::string& filename) {
    const std::string current_path = std::filesystem::current_path();
    const std::string full_path = current_path + "/../../res/" + filename;
    return CImg<unsigned char>().get_load_jpeg(&full_path[0]);
}

static CImg<unsigned char> saveImageToRes(const std::string& filename, const CImg<unsigned char>& image) {
    const std::string current_path = std::filesystem::current_path();
    const std::string full_path = current_path + "/../../res/" + filename;
    return image.save_jpeg(&full_path[0]);
}

static std::vector<std::tuple<unsigned char,unsigned char,unsigned char>> convertImageToVectorRGB(const CImg<unsigned char>& image) {
    const int width = image.width();
    const int height = image.height();
    std::vector<std::tuple<unsigned char,unsigned char,unsigned char>> res;
    for (int r = 0; r < height; ++r)
        for (int c = 0; c < width; ++c) {
            unsigned char reg = image(c, r, 0, 0),
                green = image(c, r, 0, 1),
                blue = image(c, r, 0, 2);
            res.emplace_back(reg, green, blue);
        }
    return res;
}

static CImg<unsigned char> loadImageFromVectorRGB(const std::vector<std::tuple<unsigned char,unsigned char,unsigned char>> &input, int width, int height) {
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

static void just_for_fun() {
    auto img = loadImageFromRes("exm.jpg");
    auto vec = convertImageToVectorRGB(img);
    auto imgnew = loadImageFromVectorRGB(vec, img.width(), img.height());
    saveImageToRes("exm2.jpg", imgnew);
    std::cout << "Hello, World!" << std::endl;
}
#endif