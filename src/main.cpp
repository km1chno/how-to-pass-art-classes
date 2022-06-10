#include <iostream>
#include <fstream>
#include <vector>
//#include "images/datasetManager.h"
//#include "pca/pca.h"
//#include "svm/svm.h"
#include "hog/hog.h"

using std::vector;

const std::string genresForInput[] =
        {"Post-Impressionism", "Cubism"};

vector<vector<vector<vector<float>>>> loadGenreRepresentation() {
    vector<vector<vector<vector<float>>>> allGenre;
    for (auto name : genresForInput) {
        std::ifstream in("../../res/" + name + "/_Representation");
        int n_arts;
        in >> n_arts;
        vector<vector<vector<float>>> arts;
        int width, height;
        in >> width >> height;
        for (int i = 0; i < n_arts; ++i) {
            vector <vector<float>> genre;
            for (int i = 0; i < height; ++i) {
                genre.emplace_back();
                for (int j = 0; j < width; ++j) {
                    short red, green, blue;
                    in >> red >> green >> blue;
                    genre[i].push_back((red + green + blue) / 3.0);
                }
            }
            arts.push_back(genre);
        }
        allGenre.push_back(arts);
    }
    return allGenre;
}

int main() {
    //Eigen::MatrixXf m(2,2);
    //resizeAllImagesFromRes();
    //savefloatRepresentedGenres();
    auto genresRepresentation = loadGenreRepresentation();
    auto expImageRepresentation = genresRepresentation[0][0];
    auto hog = Hog();
    auto res = hog.getHistogram(expImageRepresentation);
    std::cout << "start";
    //getCovarianceMatrix(0);
}
