#include <iostream>
#include <fstream>
#include <vector>
#include "images/datasetManager.h"

const std::string genresForInput[] =
        {"Post-Impressionism", "Cubism"};

std::vector<std::vector<std::vector<double>>> getDoubleVectorGenres() {
    std::vector<std::vector<std::vector<double>>> allGenre;
    for (const auto& name : genresForInput) {
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
    freopen("../../res/paintings.data", "w", stdout);
    std::cerr << "start1" << std::endl;
    resizeAllImagesFromRes(50, 50);
    std::cerr << "start2" << std::endl;
    saveDoubleRepresentedGenres();
    std::vector<std::vector<std::vector<double>>> vec = getDoubleVectorGenres();
    std::cerr << "start3" << "  " << vec.size() << std::endl;
    for (int i = 0; i < vec[0].size(); i++) {
        for (auto x : vec[0][i])
            std::cout << x << ",";
        std::cout << -1 << "\n";
    }
    std::cerr << "start4" << std::endl;
    for (int i = 0; i < vec[1].size(); i++) {
        for (auto x : vec[1][i])
            std::cout << x << ",";
        std::cout << 1 << "\n";
    }
    std::cerr << "start4" << std::endl;
}
