#ifndef HOW_TO_PASS_ART_CLASSES_TREE_H
#define HOW_TO_PASS_ART_CLASSES_TREE_H
#include <iostream>
#include <vector>
#include "treeNode.h"
#include <Eigen/Dense>
using std::vector;

class DecisionTree {
    double _get_entropy(const vector<int> &Y);

    std::pair<vector<int>, vector<int>> _split_children(const vector<double> &X, double cut);

    double _get_entropy_for_cut(const vector<double> &X, const vector<int> &Y, double cut);

    std::tuple<int, double, double> _find_cut(const vector<vector<double>> &X, const vector<int> &Y, const vector<int> &chosen_features);

    TreeNode* _build_tree(const vector<vector<double>> &X, const vector<int> &Y, int depth = 0);

    double _find_proper_value(const Eigen::VectorXd &x, TreeNode *actual_node);

    double _selectMostCommonValue(const vector<int> &Y);

    vector<double> _extractColumn(const vector<vector<double>> &X, int feature_number);

    vector<vector<double>> _extractRows(const vector<vector<double>> &X, const vector<int> &sample_number);

    vector<int> _extractRows1d(const vector<int> &Y, const vector<int> &sample_number);

    vector<double> _getUnique(vector<double> &&v);

    vector<int> _random_choises(int max, int amount);
public:
    int min_node_rows, max_depth;
    TreeNode *root;
    explicit DecisionTree(int _max_depth = 107, int _min_node_rows = 5) : max_depth(_max_depth), min_node_rows(_min_node_rows){}
    ~DecisionTree() {
        delete root;
    }
    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &Y);

    vector<int> predictRows(const Eigen::MatrixXd &X);
    int predict(const Eigen::VectorXd &X);

};


#endif //HOW_TO_PASS_ART_CLASSES_TREE_H
