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

/* path relative to PROJECT/res/Dataset/ without extension, example: */
/* for Dataset/Cubism/Picasso1.jpg, path = Cubism/Picasso1 */
/* copies image from Dataset to AugmentedDataset, scales the result */
void copyImageIntoAugmentedDataset(std::string &path, int width, int height);
/* generates three flipped images (x, y, xy) from original image and saves in AugmentedDataset directory, scales the result) */
void generateFlippedImages(const std::string& path, int width, int height);
/* generates rotated versions of input image (90, 180, 270) and saves in AugmentedDateset directory, scales the result) */
void generateRotatedImages(const std::string& path, int width, int height);
/* generates cropped (randomly) versions of input image and saves in AugmentedDateset directory, scales the result) */
void generateCroppedImages(const std::string& path, int width, int height, int n_images);
/* generates version of image with salt and pepper noise (random white and black pixels) and saves in AugmentedDateset directory, scales the result) */
/* intensity must be in range [0, 1] */
void generateSaltAndPepperNoiseImage(const std::string& path, int width, int height, double intensity);

/* path relative to PROJECT/res/AugmentedDataset/ without extension, example: */
/* for AugmentedDataset/Cubism/Picasso1.jpg, path = Cubism/Picasso1 */
/* generates grey scale image and saves to GreyScaleDataset directory */
void generateGrayScaleImage(const std::string& path);

#endif