//
// Created by dandrozavr on 09/06/22.
//

#include "hog.h"

#include <iostream>
#include <cmath>
#include <vector>
#include <numeric>
#include <math.h>

using std::vector;

Hog::Hog(int _cellSize, int _blockSize, int _n_bins) : n_pixels_per_cel(_cellSize), n_cells_per_block(_blockSize), n_bins(_n_bins){

}

vector<vector<float>> Hog::getHistogram(const vector<vector<float>> &image) {
    int n_rows_image = image.size();
    if (n_rows_image == 0) {
        throw "getHistogram: image is empty";
    }
    int n_cols_image = image[0].size();
    vector<vector<float>> angles(n_rows_image), magnitudes(n_rows_image);

    for (int i = 0; i < n_rows_image; ++i) {
        angles[i].reserve(n_cols_image);
        magnitudes[i].reserve(n_cols_image);
        for (int j = 0; j < n_cols_image; ++j) {
            float gradx, grady;
            computeGradient(image, i, j, gradx, grady, n_rows_image, n_cols_image);
            float angle = std::abs(std::atan2(gradx, grady));
            float magnitude = std::sqrt(gradx * gradx + grady * grady);
            angles[i].push_back(angle);
            magnitudes[i].push_back(magnitude);
        }
    }
    vector<vector<float>> answer;
    const int pixels_per_block = n_pixels_per_cel * n_cells_per_block;
    for (int block_x = 0; block_x <= n_rows_image - pixels_per_block; block_x += pixels_per_block)
        for (int block_y = 0; block_y <= n_cols_image - pixels_per_block; block_y += pixels_per_block) {
            vector<vector<float>> cellsHist;
            for (int r = block_x; r < block_x + pixels_per_block; r += n_pixels_per_cel)
                for (int c = block_y; c < block_y + pixels_per_block; c += n_pixels_per_cel) {
                    cellsHist.push_back(getCellHistogram(angles, magnitudes, r, c));
                }
            answer.push_back(l2blockNormalization(cellsHist));
        }
    return answer;
}

void Hog::computeGradient(const vector<vector<float>> &image, const int &posx, const int &posy, float &gradx, float &grady, const int &n_rows, const int &n_cols) {
    if (posx + 1 != n_cols) {
        gradx = image[posx + 1][posy];
    } else {
        gradx = 0;
    }
    if (posy + 1 != n_rows) {
        grady = image[posx][posy + 1];
    } else {
        grady = 0;
    }
    if (posx) {
        gradx -= image[posx - 1][posy];
    }
    if (posy) {
        grady -= image[posx][posy - 1];
    }
}

vector<float> Hog::getCellHistogram(const vector<vector<float>> &angles, const vector<vector<float>> &magn, const int &fromX, const int &fromY) {
    std::vector <float> bins(n_bins);
    const float bin_width = 3.145926 / n_bins;
    for (int i = 0; i < n_pixels_per_cel; ++i)
        for (int j = 0; j < n_pixels_per_cel; ++j) {
            int binIndex = angles[fromX + i][fromY + j] / bin_width;
            if (binIndex < 0)
                throw "histogram = " + std::to_string(binIndex);
            bins[binIndex] += magn[fromX + i][fromY + j];
        }
    return bins;


    // This is another (and much more advanced) option, described in towardscience. Should be tested somehow
    for (auto &i : bins) i = 0;
    for (int i = 0; i < n_pixels_per_cel; ++i)
        for (int j = 0; j < n_pixels_per_cel; ++j) {
            float fixedValue = angles[fromX + i][fromY + j] / bin_width - 0.5f;
            int binIndex = int(trunc(fixedValue) + 0.000000001);
            bins[binIndex] += magn[fromX + i][fromY + j] * fixedValue;
            float center = bin_width * (binIndex + 0.5f);
            if (binIndex + 1 < n_bins) {
                bins[binIndex + 1] += magn[fromX + i][fromY + j] * ((angles[fromX + i][fromY + j] - center) / bin_width);
            } else {
                // what should we do here? probably nothing
            }
        }
}

vector<float> Hog::l2blockNormalization(const vector<vector<float>>& vec) {
    vector <float> flat_vec, temp;
    for (const auto &v : vec)
        flat_vec.insert(std::end(flat_vec), std::begin(v), std::end(v));
    temp = flat_vec;

    std::transform(std::begin(flat_vec), std::end(flat_vec), std::end(temp), [](const float &x) {
        return x * x;
    });


    const float epsilon = 0.00000001;

    //float den = std::accumulate(std::begin(temp), std::end(temp), 0.0f);
    float den = 1;
    den = std::sqrt(den + epsilon);

    /*if (den != 0)
        std::transform(std::begin(flat_vec), std::end(flat_vec), std::begin(flat_vec), [den](const float &nom) {
            return nom / den;
        });*/
    return flat_vec;
}



