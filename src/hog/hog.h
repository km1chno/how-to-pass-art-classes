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
    vector<double> getCellHistogram(const vector<vector<double>> &angles, const vector<vector<double>> &magn, const int &fromX, const int &fromY);
    vector<double> l2blockNormalization(const vector<vector<double>>& cellsHistogram);
    void computeGradient(const vector<vector<double>> &image, const int &posx, const int &posy, double &gradx, double &grady, const int &n_rows, const int &n_cols);
public:
    Hog(int n_pixels_per_cel = 8, int n_cells_per_block = 1, int n_bins = 9);

    vector<vector<double>> getHistogram(const vector<vector<double>> &image);
    vector<double> getFlatHistogram(const vector<vector<double>> &image);

};


#endif //HOW_TO_PASS_ART_CLASSES_HOG_H
