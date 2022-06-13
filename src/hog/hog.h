//
// Created by dandrozavr on 09/06/22.
//

#ifndef HOW_TO_PASS_ART_CLASSES_HOG_H
#define HOW_TO_PASS_ART_CLASSES_HOG_H

#include <vector>

using std::vector;

class Hog {
    int n_pixels_per_cel, n_cells_per_block;
    int n_bins; // 20 degrees
    vector<float> getCellHistogram(const vector<vector<float>> &angles, const vector<vector<float>> &magn, const int &fromX, const int &fromY);
    vector<float> l2blockNormalization(const vector<vector<float>>& cellsHistogram);
    void computeGradient(const vector<vector<float>> &image, const int &posx, const int &posy, float &gradx, float &grady, const int &n_rows, const int &n_cols);
public:
    Hog(int n_pixels_per_cel = 8, int n_cells_per_block = 1, int n_bins = 9);

    vector<vector<float>> getHistogram(const vector<vector<float>> &image);
    vector<float> getFlatHistogram(const vector<vector<float>> &image);

};


#endif //HOW_TO_PASS_ART_CLASSES_HOG_H
