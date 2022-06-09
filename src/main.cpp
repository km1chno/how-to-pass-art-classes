#include <iostream>
#include <fstream>
#include <vector>
//#include "images/datasetManager.h"
//#include "pca/pca.h"
//#include "svm/svm.h"

const std::string genresForInput[] =
        {"Post-Impressionism", "Cubism"};

std::vector<std::vector<std::vector<double>>> getDoubleVectorGenres() {
    std::vector<std::vector<std::vector<double>>> allGenre;
    for (auto name : genresForInput) {
        std::ifstream in("../../res/" + name + "/_Representation");
        int n_arts;
        in >> n_arts;
        std::vector<std::vector<double>> arts;
        int width, height;
        in >> width >> height;
        for (int i = 0; i < n_arts; ++i) {
            std::vector <double> genre;
            for (int i = 0; i < width * height; ++i) {
                short red, green, blue;
                in >> red >> green >> blue;
                genre.push_back((red + green + blue) / 3.0);
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
    //saveDoubleRepresentedGenres();
    //getDoubleVectorGenres();
    std::cout << "start";
    //getCovarianceMatrix(0);
}
