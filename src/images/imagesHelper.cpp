#include <iostream>
#include <filesystem>
#define cimg_use_jpeg
#include <CImg-3.1.3/CImg.h>
#include <random>
#include "imagesHelper.h"

using namespace cimg_library;

typedef std::tuple<unsigned char,unsigned char,unsigned char> unsignedCharTuple;

CImg<unsigned char> resizeImage(const CImg<unsigned char> &image, int width, int height) {
    return std::move(image.get_resize(width, height));
}

std::string getResPath() {
    return std::string(std::filesystem::current_path()) + "/../res/";
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

void copyImageIntoAugmentedDataset(std::string &path, int width, int height) {
    std::string filePath = "Dataset/" + path + ".jpg";
    auto img = loadImageFromRes(filePath);
    saveImageToRes("AugmentedDataset/" + path + "_orig.jpg", resizeImage(img, width, height));
}

void generateFlippedImages(const std::string& path, int width, int height) {
    std::string filePath = "Dataset/" + path + ".jpg";
    auto img = loadImageFromRes(filePath);
    img.mirror('x');
    saveImageToRes("AugmentedDataset/" + path + "_flip_x.jpg", resizeImage(img, width, height));
    img.mirror("xy");
    saveImageToRes("AugmentedDataset/" + path + "_flip_y.jpg", resizeImage(img, width, height));
    img.mirror("x");
    saveImageToRes("AugmentedDataset/" + path + "_flip_xy.jpg", resizeImage(img, width, height));
}

void generateRotatedImages(const std::string& path, int width, int height) {
    std::string filePath = "Dataset/" + path + ".jpg";
    auto img = loadImageFromRes(filePath);
    img.rotate(90);
    saveImageToRes("AugmentedDataset/" + path + "_rot_90.jpg", resizeImage(img, width, height));
    img.rotate(90);
    saveImageToRes("AugmentedDataset/" + path + "_rot_180.jpg", resizeImage(img, width, height));
    img.rotate(90);
    saveImageToRes("AugmentedDataset/" + path + "_rot_270.jpg", resizeImage(img, width, height));
}

void generateCroppedImages(const std::string& path, int width, int height, int n_images) {
    std::string filePath = "Dataset/" + path + ".jpg";
    auto img = loadImageFromRes(filePath);
    int w = img.width();
    int h = img.height();

    /* x0 \in [0, w/3]
     * x1 \in [w-w/3, w-1]
     * y0 \in [0, h/3]
     * y1 \in [h-h/3, h-1]
     */
    std::random_device rd;
    std::mt19937 gen(rd());
    std::vector<std::uniform_int_distribution<>> dist;
    dist.emplace_back(1, w/3);
    dist.emplace_back(w-w/3, w-1);
    dist.emplace_back(1, h/3);
    dist.emplace_back(h-h/3, h-1);

    for (int i = 0; i < n_images; i++) {
        auto newImg = img;
        newImg.crop(dist[0](gen), dist[1](gen), dist[2](gen), dist[3](gen));
        saveImageToRes("AugmentedDataset/" + path + "_crop_" + std::to_string(i) + ".jpg", resizeImage(newImg, width, height));
    }
}

void generateSaltAndPepperNoiseImage(const std::string& path, int width, int height, double intensity) {
    std::string filePath = "Dataset/" + path + ".jpg";
    auto img = loadImageFromRes(filePath);
    int w = img.width();
    int h = img.height();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distx(0, w-1);
    std::uniform_int_distribution<> disty(0, h-1);
    std::uniform_int_distribution<> bidist(0, 1);

    double pixelsToChange = intensity * double(w*h);
    int color[2][3] = {{0, 0, 0}, {255, 255, 255}};
    for (int i = 0; i < int(pixelsToChange); i++) {
        int x = distx(gen);
        int y = disty(gen);
        int c = bidist(gen);
        img.draw_point(x, y, color[c]);
    }

    saveImageToRes("AugmentedDataset/" + path + "_noise.jpg", resizeImage(img, width, height));
}

void generateGrayScaleImage(const std::string &path) {
    std::string filePath = "AugmentedDataset/" + path + ".jpg";
    auto img = loadImageFromRes(filePath);
    for (int i = 0; i < img.width(); i++)
        for (int j = 0; j < img.height(); j++) {
            int r = int(img(i, j, 0));
            int g = int(img(i, j, 1));
            int b = int(img(i, j, 2));
            int greyPixel = (r + g + b) / 3;

            unsigned char color[3] = {(unsigned char)greyPixel, (unsigned char)greyPixel, (unsigned char)greyPixel};
            img.draw_point(i, j, color);
        }
    saveImageToRes("GreyScaleDataset/" + path + ".jpg", img);
}
