#ifndef _TOOLS_
#define _TOOLS_

#include <Eigen/Dense>

/* standarizes data in all three matrices using means and variances from A */
void standarizeData(Eigen::MatrixXd &A, Eigen::MatrixXd &B, Eigen::MatrixXd &C);

/* scales data in all three matrices into range [0, 1] using min max method on matrix A */
void minmaxscaling(Eigen::MatrixXd &A, Eigen::MatrixXd &B, Eigen::MatrixXd &C);

/* reads data from .csv file (path) and returns as 2D vector of floats */
std::vector<std::vector<double> > inputFromCSV(const std::string& path);

/* splits rows in data by its class (first column) into target[] vectors, only uses fraction of rows in each class */
void splitDataByClass(std::vector<std::vector<double> > &data, double fraction, std::vector<std::vector<double> > target[], int N_CLASSES);

/* returns (X_train, X_val, X_test) and (y_train, y_val, y_test) */
std::pair<std::vector<Eigen::MatrixXd>, std::vector<Eigen::VectorXd> > inputAndPrepareDataForMulticlassClassification(
        const std::string &file_path,
        double data_fraction,
        std::set<std::string> &params,
        int N_CLASSES,
        double TRAIN_FRACTION,
        double VAL_FRACTION,
        double PCA_FRACTION
);

/* returns (X_train, X_val) and (y_train, y_val) for classes i, j */
std::pair<std::vector<Eigen::MatrixXd>, std::vector<Eigen::VectorXd> > extractDataForBinaryClassification(
        Eigen::MatrixXd &X_train,
        Eigen::MatrixXd &X_val,
        Eigen::VectorXd &y_train,
        Eigen::VectorXd &y_val,
        int i,
        int j
);

#endif