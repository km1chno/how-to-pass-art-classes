#ifndef _IMAGES_HELPER_
#define _IMAGES_HELPER_

#define cimg_use_jpeg

#include <string>
#include <filesystem>
#include <vector>
#include <CImg-3.1.3/CImg.h>

using namespace cimg_library;

typedef std::tuple<unsigned char,unsigned char,unsigned char> unsignedCharTuple;

CImg<unsigned char> resizeImage(const CImg<unsigned char> &image, int width, int height);
std::string getResPath();
CImg<unsigned char> loadImageFromRes(const std::string& filename);
CImg<unsigned char> saveImageToRes(const std::string& filename, const CImg<unsigned char>& image);
std::vector<unsignedCharTuple> convertImageToVectorRGB(const CImg<unsigned char>& image);
CImg<unsigned char> loadImageFromVectorRGB(const std::vector<unsignedCharTuple> &input, int width, int height);
void just_for_fun();

#endif