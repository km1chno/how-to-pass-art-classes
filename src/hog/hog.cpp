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

vector<vector<double>> Hog::getHistogram(const vector<vector<double>> &image) {
    int n_rows_image = image.size();
    if (n_rows_image == 0) {
        throw "getHistogram: image is empty";
    }
    int n_cols_image = image[0].size();
    vector<vector<double>> angles(n_rows_image), magnitudes(n_rows_image);

    for (int i = 0; i < n_rows_image; ++i) {
        angles[i].reserve(n_cols_image);
        magnitudes[i].reserve(n_cols_image);
        for (int j = 0; j < n_cols_image; ++j) {
            double gradx, grady;
            computeGradient(image, i, j, gradx, grady, n_rows_image, n_cols_image);
            double angle = std::abs(std::atan2(gradx, grady));
            double magnitude = std::sqrt(gradx * gradx + grady * grady);
            angles[i].push_back(angle);
            magnitudes[i].push_back(magnitude);
        }
    }
    vector<vector<double>> answer;
    const int pixels_per_block = n_pixels_per_cel * n_cells_per_block;
    for (int block_x = 0; block_x <= n_rows_image - pixels_per_block; block_x += pixels_per_block)
        for (int block_y = 0; block_y <= n_cols_image - pixels_per_block; block_y += pixels_per_block) {
            vector<vector<double>> cellsHist;
            for (int r = block_x; r < block_x + pixels_per_block; r += n_pixels_per_cel)
                for (int c = block_y; c < block_y + pixels_per_block; c += n_pixels_per_cel) {
                    cellsHist.push_back(getCellHistogram(angles, magnitudes, r, c));
                }
            answer.push_back(l2blockNormalization(cellsHist));
        }
    return answer;
}

vector<double> Hog::getFlatHistogram(const vector<vector<double>> &image) {
    auto tempRes = getHistogram(image);
    vector<double> res;
    for (const auto &vec : tempRes) {
        res.reserve(res.size() + vec.size());
        res.insert(std::end(res), std::begin(vec), std::end(vec));
    }
    return res;
}

void Hog::computeGradient(const vector<vector<double>> &image, const int &posx, const int &posy, double &gradx, double &grady, const int &n_rows, const int &n_cols) {
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

vector<double> Hog::getCellHistogram(const vector<vector<double>> &angles, const vector<vector<double>> &magn, const int &fromX, const int &fromY) {
    std::vector <double> bins(n_bins);
    const double bin_width = 3.145926 / n_bins;
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
            double fixedValue = angles[fromX + i][fromY + j] / bin_width - 0.5f;
            int binIndex = int(trunc(fixedValue) + 0.000000001);
            bins[binIndex] += magn[fromX + i][fromY + j] * fixedValue;
            double center = bin_width * (binIndex + 0.5f);
            if (binIndex + 1 < n_bins) {
                bins[binIndex + 1] += magn[fromX + i][fromY + j] * ((angles[fromX + i][fromY + j] - center) / bin_width);
            } else {
                // what should we do here? probably nothing
            }
        }
}

vector<double> flatter(const vector<vector<double>> &vec) {
    vector<double> flat_vec;
    for (const auto &v : vec) {
        flat_vec.reserve(flat_vec.size() + v.size() + 1);
        flat_vec.insert(std::end(flat_vec), std::begin(v), std::end(v));
    }
    return std::move(flat_vec);
}

vector<double> Hog::l2blockNormalization(const vector<vector<double>>& vec) {
    vector <double> flat_vec = flatter(vec), temp;
    temp = flat_vec;
    std::transform(std::begin(flat_vec), std::end(flat_vec), std::begin(temp), [](const double &x) {
        return x * x;
    });
    const double epsilon = 0.00000001;

    double den = std::accumulate(std::begin(temp), std::end(temp), 0.0f);
    den = std::sqrt(den + epsilon);

    if (den != 0)
        std::transform(std::begin(flat_vec), std::end(flat_vec), std::begin(flat_vec), [den](const double &nom) {
            return nom / den;
        });
    return std::move(flat_vec);
}



