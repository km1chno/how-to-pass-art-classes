#ifndef _DATASET_MANAGER_
#define _DATASET_MANAGER_

#include <iostream>
#include <fstream>
#include "imagesHelper.h"

using namespace cimg_library;

void resizeAllImagesFromRes(int width=100, int height=100);
void saveDoubleRepresentedSingleImage(const std::vector<unsignedCharTuple>& imageVector, std::ofstream &out);
void saveDoubleRepresentedGenres();

#endif