#ifndef HOW_TO_PASS_ART_CLASSES_GLCM_H
#define HOW_TO_PASS_ART_CLASSES_GLCM_H

#include <Eigen/Dense>
using std::vector;

class Glcm {
    int offsetx, offsety, n_tones;
    int n_rows, n_cols;
    int bin_width;
    Eigen::MatrixXd matrix;
    int getTone(const double &pixel);
    double _informationMeasuresofCorrelation(int type);
public:
    Glcm(int offsetx, int offsety, int n_tones);
    void computeMatrix(const vector<vector<double>> &image);
    vector<double> getFlatMatrix();
    vector<double> getAllFeaturesFromMatrix();
    double mean();
    double std();
    double contrast();
    double dissimilarity();
    double homogeneity();
    double ASM();
    double energy();
    double max();
    double correlation();
    double variance();
    double inverseDifferenceMoment();
    double sumAverage();;
    double entropy();
    double sumEntropy();
    double diffVariance();
    double diffEntropy();
    double informationMeasuresofCorrelation1();
    double informationMeasuresofCorrelation2();
};

vector<vector<double>> useGlcmFeatures(const vector<vector<double>> &vt);

#endif //HOW_TO_PASS_ART_CLASSES_GLCM_H
