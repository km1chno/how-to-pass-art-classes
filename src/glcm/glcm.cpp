#include "glcm.h"
#include <iostream>
#include <iomanip>

using std::vector;
const int max_pixel_value = 256;
const double EPSILON = 0.000000001;


Glcm::Glcm(int _offsetx, int _offsety, int _n_tones) :
    offsetx(_offsetx), offsety(_offsety), n_tones(_n_tones) {
    this->bin_width = max_pixel_value / n_tones + 1;
}

void Glcm::computeManyMatrixes(const vector<vector<double>> &image, int matrix_size) {
    int img_rows = image.size();
    int img_cols = image[0].size();
    Eigen::MatrixXd imageMatrix(img_rows, img_cols);
    for (int i = 0; i < img_rows; ++i)
        for (int j = 0; j < img_cols; ++j)
            imageMatrix(i, j) = image[i][j];

    for (int i = 0; i < img_rows; i += matrix_size)
        for (int j = 0; j < img_cols; j += matrix_size) {
            //std::cout << i << " MATRIX " << j << " " << matrix_size << " " << n_rows << " " << n_cols << '\n';
            int width = std::min(matrix_size, img_rows - i);
            int height = std::min(matrix_size, img_cols - j);
            Eigen::MatrixXd inputImage = imageMatrix.block(i, j, width, height);
            computeMatrix(inputImage);
            allMatrix.push_back(matrix);
        }
}

void Glcm::computeMatrix(const vector<vector<double>> &image) {
    this->n_rows = image.size();
    this->n_cols = image[0].size();
    Eigen::MatrixXd imageMatrix(n_rows, n_cols);
    for (int i = 0; i < n_rows; ++i)
        for (int j = 0; j < n_cols; ++j)
            imageMatrix(i, j) = image[i][j];
    computeMatrix(imageMatrix);
}

void Glcm::computeMatrix(const Eigen::MatrixXd &imageMatrix) {
    this->matrix = Eigen::MatrixXd(n_tones, n_tones);
    this->n_rows = imageMatrix.rows();
    this->n_cols = imageMatrix.cols();
    if (n_rows == 0) {
        throw "getHistogram: image is empty";
    }
    int count = 0; // normalizing factor
    for (int r = 0; r < n_tones; ++r)
        for (int c = 0; c < n_tones; ++c) {
            matrix(r, c) = 0;
        }

    for (int r = 0; r < n_rows; ++r)
        for (int c = 0; c < n_cols; ++c) {
            int x = r + offsetx;
            int y = c + offsety;
            if (x >= 0 && y >= 0 && x < n_rows && y < n_cols) {
                int first_tone = getTone(imageMatrix(r, c));
                int second_tone = getTone(imageMatrix(x, y));
                ++matrix(first_tone, second_tone);
                ++matrix(second_tone, first_tone);
                count += 2;
            }
        }
    if (count > 0)
        for (int ftone = 0; ftone < n_tones; ++ftone)
            for (int stone = 0; stone < n_tones; ++stone) {
                matrix(ftone, stone) /= count;
                //std::cout << std::setprecision(5) << matrix(ftone, stone) << " ";
            }
    //std::cout << '\n';
}

inline int Glcm::getTone(const double &pixel) {
    return pixel / bin_width;
}

vector<double> Glcm::getFlatMatrix() {
    vector<double> result;
    result.reserve(n_tones * n_tones);
    for (int i = 0; i < n_tones; ++i)
        for (int j = 0; j < n_tones; ++j)
            result.push_back(matrix(i, j));
    return result;
}

double Glcm::mean() {
    double mean = 0;
    const int n_tonesSqr = n_tones * n_tones;
    for (int r = 0; r < n_tones; ++r)
        for (int c = 0; c < n_tones; ++c)
            mean += matrix(r, c) * r / n_tonesSqr;
    return mean;
}

double Glcm::max() {
    double answer = 0;
    for (int r = 0; r < n_tones; ++r)
        for (int c = 0; c < n_tones; ++c)
            answer = std::max(answer, matrix(r, c));
    return answer;
}

double Glcm::std() {
    double mean_ = mean();
    double stdsqr = 0;
    for (int r = 0; r < n_tones; ++r)
        for (int c = 0; c < n_tones; ++c) {
            double temp = matrix(r, c) * r - mean_;
            stdsqr += temp * temp;
        }
    return std::sqrt(stdsqr);
}

/** Contrast
* The contrast feature is a difference moment of the P matrix and is a
* measure of the contrast or the amount of local variations present in an
* image.
*/

double Glcm::contrast() {
    double answer = 0;
    for (int r = 0; r < n_tones; ++r)
        for (int c = 0; c < n_tones; ++c)
            answer += matrix(r, c) * (r - c) * (r - c);
    return answer;
}

double Glcm::dissimilarity() {
    double answer = 0;
    for (int r = 0; r < n_tones; ++r)
        for (int c = 0; c < n_tones; ++c)
            answer += matrix(r, c) * abs(r - c);
    return answer;
}

double Glcm::homogeneity() {
    double answer = 0;
    for (int r = 0; r < n_tones; ++r)
        for (int c = 0; c < n_tones; ++c)
            answer += matrix(r, c) / (1.0 + (r - c) * (r - c));
    return answer;
}

double Glcm::energy() {
    return std::sqrt(ASM());
}

/** Angular Second Moment
* The angular second-moment feature (ASM) f1 is a measure of homogeneity
* of the image. In a homogeneous image, there are very few dominant
* gray-tone transitions. Hence the P matrix for such an image will have
* fewer entries of large magnitude.
*/

double Glcm::ASM() {
    double sum = 0;
    for (int r = 0; r < n_tones; ++r)
        for (int c = 0; c < n_tones; ++c)
            sum += matrix(r, c) * matrix(r, c);
    return sum;
}

double Glcm::_informationMeasuresofCorrelation(int type) {
    vector <double> px(n_tones, 0), py(n_tones, 0);
    for (int i = 0; i < n_tones; ++i)
        for (int j = 0; j < n_tones; ++j) {
            px[j] += matrix(i, j);
            py[j] += matrix(i, j);
        }
    double hxy = 0, hxy1 = 0, hxy2 = 0, hx = 0, hy = 0;
    for (int i = 0; i < n_tones; ++i)
        for (int j = 0; j < n_tones; ++j) {
            hxy -= matrix(i, j) * log10(matrix(i, j) + EPSILON)/log10(2.0);
            hxy1 -= matrix(i, j) * log10(px[i] * py[j] + EPSILON)/log10(2.0);
            hxy2 -= px[i] * py[j] * log10(px[i] * py[j] + EPSILON)/log10(2.0);
        }
    for (int i = 0; i < n_tones; ++i) {
        hx -= px[i] * log10 (px[i] + EPSILON)/log10(2.0);
        hy -= py[i] * log10 (py[i] + EPSILON)/log10(2.0);
    }
    if (type == 1) {
        if ((hx > hy ? hx : hy)==0)
            return 1;
        return ((hxy - hxy1) / (hx > hy ? hx : hy));
    }
    return (sqrt(fabs(1 - exp (-2.0 * (hxy2 - hxy)))));
}

double Glcm::informationMeasuresofCorrelation1() {
    return _informationMeasuresofCorrelation(1);
}

double Glcm::informationMeasuresofCorrelation2() {
    return _informationMeasuresofCorrelation(2);
}

/** Correlation
*
* This correlation feature is a measure of gray-tone linear-dependencies
* in the image.
*/

double Glcm::correlation() {
    vector <int> px(n_tones, 0);
    double meanx = 0, meany = 0, sum_sqrx = 0, sum_sqry = 0, tmp = 0;
    for (int i = 0; i < n_tones; ++i)
        for (int j = 0; j < n_tones; ++j)
            px[i] += matrix(i, j);
    for (int i = 0; i < n_tones; ++i) {
        meanx += px[i] * i;
        sum_sqrx += px[i] * i * i;
    }
    meany = meanx;
    sum_sqry = sum_sqrx;
    double stddevx = sqrt (sum_sqrx - (meanx * meanx));
    double stddevy = stddevx;
    tmp = 0;
    /* Finally, the correlation ... */
    for (int i = 0; i < n_tones; ++i)
        for (int j = 0; j < n_tones; ++j)
            tmp += i * j * matrix(i, j);
    if (stddevx * stddevy==0)
        return 1;
    return (tmp - meanx * meany) / (stddevx * stddevy);
}



/** Sum of Squares: Variance
* Calculates the mean intensity level instead of the mean of
*  cooccurrence matrix elements
* */

double Glcm::variance() {
    double mean = 0, var = 0;
    for (int i = 0; i < n_tones; ++i)
        for (int j = 0; j < n_tones; ++j)
            mean += i * matrix(i, j);
    for (int i = 0; i < n_tones; ++i)
        for (int j = 0; j < n_tones; ++j)
            var += (i - mean) * (i - mean) * matrix(i, j);
    return var;
}

double Glcm::inverseDifferenceMoment() {
    double result = 0;
    for (int i = 0; i < n_tones; ++i)
        for (int j = 0; j < n_tones; ++j)
            result += matrix(i, j) / (1 + (i - j) * (i - j));
    return result;
}

double Glcm::sumAverage() {
    double savg = 0;
    vector <double> Pxpy(n_tones * 2, 0);
    for (int i = 0; i < n_tones; ++i)
        for (int j = 0; j < n_tones; ++j)
            Pxpy[i + j] += matrix(i, j);
    for (int i = 0; i <= (2 * n_tones - 2); ++i)
        savg += i * Pxpy[i];
    return savg;
}

double Glcm::entropy() {
    double entropy = 0;
    for (int i = 0; i < n_tones; ++i)
        for (int j = 0; j < n_tones; ++j)
            entropy += matrix(i, j) * log10(matrix(i, j) + EPSILON) / log10(2.0);
    return -entropy;
}

double Glcm::sumEntropy() {
    double sentropy = 0;
    vector <double> Pxpy(n_tones * 2, 0);
    for (int i = 0; i < n_tones; ++i)
        for (int j = 0; j < n_tones; ++j)
            Pxpy[i + j] += matrix(i, j);
    for (int i = 2; i < n_tones * 2; ++i)
        sentropy -= Pxpy[i] * log10(Pxpy[i] + EPSILON) / log10(2.0);
    return -sentropy;
}

double Glcm::diffVariance() {
    double sum = 0, sum_sqr = 0, var = 0;
    vector <double> Pxpy(n_tones, 0);
    for (int i = 0; i < n_tones; ++i)
        for (int j = 0; j < n_tones; ++j)
            Pxpy[abs(i - j)] += matrix(i, j);
    for (int i = 0; i < n_tones; ++i) {
        sum += i * Pxpy[i] ;
        sum_sqr += i * i * Pxpy[i] ;
    }
    var = sum_sqr - sum * sum;
    return var;
}

double Glcm::diffEntropy() {
    double sum = 0;
    vector <double> Pxpy(n_tones, 0);
    for (int i = 0; i < n_tones; ++i)
        for (int j = 0; j < n_tones; ++j)
            Pxpy[abs(i - j)] += matrix(i, j);
    for (int i = 0; i < n_tones; ++i)
        sum += Pxpy[i] * log10(Pxpy[i] + EPSILON) / log10(2.0);
    return -sum;
}

vector<double> Glcm::getAllFeaturesFromMatrix() {
    vector<double> features;
    for (const auto &i : allMatrix) {
        n_rows = i.rows();
        n_cols = i.cols();
        matrix = i;
        features.push_back(mean());
        features.push_back(std());
        features.push_back(contrast());
        features.push_back(dissimilarity());
        features.push_back(homogeneity());
        features.push_back(ASM());
        features.push_back(energy());
        features.push_back(max());
        features.push_back(correlation());
        features.push_back(variance());
        features.push_back(inverseDifferenceMoment());
        features.push_back(sumAverage());
        features.push_back(entropy());
        features.push_back(sumEntropy());
        features.push_back(diffVariance());
        features.push_back(diffEntropy());
        features.push_back(informationMeasuresofCorrelation1());
        features.push_back(informationMeasuresofCorrelation2());
    }
    int oldFeatSize = features.size();
    for (int i = 0; i < 18; ++i) {
        double min = 1e9, max = -1e9, sum = 0;
        for (int j = i; j < oldFeatSize; ++j) {
            sum += features[j];
            min = std::min(min, features[j]);
            max = std::max(max, features[j]);
        }
        features.push_back(sum);
        features.push_back(min);
        features.push_back(max);
        features.push_back(sum / max);
    }
    //for (auto i : features) std::cout << i << " ";
//    std::cout << '\n';
    return features;
}

vector<vector<double>> useGlcmFeatures(const vector<vector<double>> &vt) {
    vector<vector<double>> newVt;
    for (const auto &vec : vt) {

        const int sz = int(sqrt(vec.size()));
        vector<vector<double>> image(sz);
        for (int i = 1; i < vec.size(); ++i) {
            image[(i-1) / sz].push_back(vec[i - 1]);
        }
        Glcm glcm = Glcm(2, 0, 7);
        Glcm glcm2 = Glcm(0, 2, 7);
        Glcm glcm3 = Glcm(2, 2, 7);

        glcm.computeManyMatrixes(image, 100);
        glcm.computeManyMatrixes(image, 50);
        //glcm.computeManyMatrixes(image, 25);
        //glcm.computeManyMatrixes(image, 5); /// MUCH MORE SLOW, AND CAN BE EVEN USELESS

        glcm2.computeManyMatrixes(image, 100);
        glcm2.computeManyMatrixes(image, 50);
        //glcm2.computeManyMatrixes(image, 25);
        //glcm2.computeManyMatrixes(image, 5); /// MUCH MORE SLOW, AND CAN BE EVEN USELESS

        glcm3.computeManyMatrixes(image, 100);
        glcm3.computeManyMatrixes(image, 50);
        //glcm3.computeManyMatrixes(image, 25);
        //glcm3.computeManyMatrixes(image, 5); /// MUCH MORE SLOW, AND CAN BE EVEN USELESS

        auto glcm_flat = glcm.getAllFeaturesFromMatrix();
        auto glcm_flat2 = glcm2.getAllFeaturesFromMatrix();
        auto glcm_flat3 = glcm3.getAllFeaturesFromMatrix();
        glcm_flat.insert(glcm_flat.end(), glcm_flat.begin(), glcm_flat.end());
        for (int i = 0; i < glcm_flat2.size(); ++i)
            glcm_flat[i] += glcm_flat2[i];// + glcm_flat3[i];

        glcm_flat.insert(glcm_flat.end(), glcm_flat2.begin(), glcm_flat2.end());
        glcm_flat.insert(glcm_flat.end(), glcm_flat3.begin(), glcm_flat3.end());
        //std::cout << glcm_flat.size() << '\n';
        //std::cout << glcm_flat[0] << " " << vec[0] << '\n';
        newVt.push_back(glcm_flat);
    }
    return newVt;
}

