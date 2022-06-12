#ifndef HOW_TO_PASS_ART_CLASSES_DATASETMANAGER_H
#define HOW_TO_PASS_ART_CLASSES_DATASETMANAGER_H

#include <iostream>
#include <fstream>
#include <vector>
typedef std::tuple<unsigned char,unsigned char,unsigned char> unsignedCharTuple;

void resizeAllImagesFromRes(int width = 100, int height = 100);

void saveDoubleRepresentedSingleImage(const std::vector<unsignedCharTuple>& imageVector, std::ofstream &out);

void saveDoubleRepresentedGenres();

/* delete all files from AugmentedDataset and GreyScaleDataset */
void deleteAugmentedDataset();
/* prepares dataset for learning (augmentation + greyscale) */
void prepareDataset(int width, int height, bool augment);

#endif //HOW_TO_PASS_ART_CLASSES_DATASETMANAGER_H
