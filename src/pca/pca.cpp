#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;

enum PCAOptions {
    useComponents = 0,
    useFraction = 1
};

/* Standarize X before using! */
MatrixXf PCA(MatrixXf X, int components=1, double infoFraction=1, PCAOptions options=useComponents) {
    MatrixXf centered = X.rowwise() - X.colwise().mean();
    MatrixXf cov = (centered.adjoint() * centered) / double(X.rows() - 1);

    SelfAdjointEigenSolver<MatrixXf> es(cov);
    std::cout << "The eigenvalues of cov matrix are:\n" << es.eigenvalues().transpose() << std::endl;
    std::cout << "Eigenvectors:\n" << es.eigenvectors() << std::endl;

    VectorXf normalizedEigenvalues = es.eigenvalues() / es.eigenvalues().sum();
    std::cout << "The normalized eigenvalues of cov matrix are:\n" << normalizedEigenvalues << std::endl;

    double eigenSum = 0;
    int _components = 0;
    for (int i = normalizedEigenvalues.cols()-1; i >= 0; i--) {
        eigenSum += normalizedEigenvalues(i);
        _components++;
        if (eigenSum >= infoFraction)
            break;
    }

    if (options == useComponents) 
        _components = components;

    MatrixXf featureMat = es.eigenvectors().rightCols(_components);

    std::cout << "Feature matrix:\n" << featureMat << "\n";

    MatrixXf result = X * featureMat;
    return result;
}

int main() {
    MatrixXf mat = MatrixXf::Random(3, 5);
    MatrixXf result = PCA(mat, 2, 0, useComponents);

    std::cout << "input:\n" << mat << "\n\nresult:\n" << result << "\n";
}