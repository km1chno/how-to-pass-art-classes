//
// Created by dandrozavr on 12/06/22.
//

#include <tuple>
#include <random>
#include <algorithm>
#include <cmath>
#include <chrono>
#include "tree.h"

double DecisionTree::_get_entropy(const vector<int> &Y) {
    int mx = *std::max_element(Y.begin(), Y.end());
    int mn = *std::min_element(Y.begin(), Y.end());
    vector <int> bins(mx - mn + 1, 0);
    for (const auto &y : Y)
        ++bins[y - mn];
    double sum = 0;
    for (auto bin : bins) {
        if (!bin) continue;
        double p = (double)(bin) / Y.size();
        sum += -p * std::log2(p);
    }
    return sum;
}

std::pair<vector<int>, vector<int>> DecisionTree::_split_children(const vector<double> &X, double cut) {
    vector<int> first, second;
    for (int i = 0; i < X.size(); ++i)
        if (X[i] <= cut) {
            first.push_back(i);
        } else {
            second.push_back(i);
        }
    return std::pair<vector<int>, vector<int>>(std::move(first), std::move(second));
}

double DecisionTree::_get_entropy_for_cut(const vector<double> &X, const vector<int> &Y, double cut) {
    auto parent_entropy = _get_entropy(Y);
    auto children_idxs = _split_children(X, cut);
    auto left_child_idxs = children_idxs.first;
    auto right_child_idxs = children_idxs.second;
    int size = Y.size(), left_size = left_child_idxs.size(), right_size = right_child_idxs.size();
    if (left_size <= min_node_rows || right_size <= min_node_rows)
        return 0;
    double left_child_entropy = ((double)(left_size) / size) * _get_entropy(_extractRows1d(Y, left_child_idxs));
    double right_child_entropy = ((double)(right_size) / size) * _get_entropy(_extractRows1d(Y, right_child_idxs));
    //std::cout << left_child_entropy << " " << right_child_entropy << " " << left_size << " " << right_size;
    return parent_entropy - left_child_entropy - right_child_entropy;
}

vector<double> DecisionTree::_getUnique(vector<double> &&v) {
    sort(v.begin(), v.end());
    v.resize(std::unique(v.begin(), v.end()) - v.begin());
    return std::move(v);
}

std::tuple<int, double, double> DecisionTree::_find_cut(const vector<vector<double>> &X, const vector<int> &Y, const vector<int> &chosen_features) {
    double opt_entropy = -1, opt_cut = -1;
    int opt_feature = -1;
    for (const int &feature : chosen_features) {
        auto newX = _getUnique(_extractColumn(X, feature));
        for (const auto &try_cut : newX) {
            auto entropy = _get_entropy_for_cut(newX, Y, try_cut);
            if (entropy == 1) break;
            if (entropy > opt_entropy) {
                opt_entropy = entropy;
                opt_cut = try_cut;
                opt_feature = feature;
            }
        }
    }
    return std::tuple<int, double, double>(opt_feature, opt_cut, opt_entropy);
}


double DecisionTree::_selectMostCommonValue(const vector<int> &Y) {
    int mx = *std::max_element(Y.begin(), Y.end());
    int mn = *std::min_element(Y.begin(), Y.end());
    vector <int> bins(mx - mn + 1, 0);
    for (const auto &y : Y)
        ++bins[y - mn];
    int max_bin_val = -1;
    int ind = -1;
    for (int i = 0; i <= mx - mn; ++i)
        if (max_bin_val < bins[i]) {
            max_bin_val = bins[i];
            ind = i;
        }
    return ind + mn;
}

vector<double> DecisionTree::_extractColumn(const vector<vector<double>> &X, int feature_number) {
    vector <double> result;
    for (const auto &x : X)
        result.push_back(x[feature_number]);
    return std::move(result);
}

vector<vector<double>> DecisionTree::_extractRows(const vector<vector<double>> &X, const vector<int> &sample_number) {
    vector <vector<double>> result;
    for (const int &ind : sample_number)
        result.push_back(X[ind]);
    return std::move(result);
}

vector<int> DecisionTree::_extractRows1d(const vector<int> &Y, const vector<int> &sample_number) {
    vector <int> result;
    for (const auto &ind : sample_number)
        result.push_back(Y[ind]);
    return std::move(result);
}

std::mt19937 gen(std::chrono::system_clock::now().time_since_epoch().count());

vector<int> DecisionTree::_random_choises(int max, int amount) {
    vector<int> choices;
    for (int i = 0; i < max; ++i)
        choices.push_back(i);
    return choices;
    // change for get later
}

TreeNode* DecisionTree::_build_tree(const vector<vector<double>> &X, const vector<int> &Y, int depth) {
    //std::cout << depth << " " << Y.size() << '\n';
    int n_samples = X.size();
    if (X.size() == 0) {
        throw "X size is 0 in decision trees";
    }
    int n_features = X[0].size();
    if (depth >= max_depth || Y.size() == 1 || min_node_rows >= n_samples) {
        double value_node = _selectMostCommonValue(Y);
        return new TreeNode(value_node, -1, -1);
    }
    for (int i = 0; i < 10; ++i) {
        if (i == 9)
            break;
        auto chosen_features = _random_choises(n_features, int(n_features * 1)); // change to variable coefficient
        int best_feature;
        double best_cut, best_entropy;
        std::tie(best_feature, best_cut, best_entropy) = _find_cut(X, Y, chosen_features);
        //std::cout << best_feature << " " << best_cut << " " << best_entropy << std::endl;
        if (best_entropy == 0)
            continue;
        auto children_idxs = _split_children(_extractColumn(X, best_feature), best_cut);
        auto left_child_idxs = children_idxs.first;
        auto right_child_idxs = children_idxs.second;
        auto left_child_node = _build_tree(_extractRows(X, left_child_idxs), _extractRows1d(Y, left_child_idxs), depth + 1);
        auto right_child_node = _build_tree(_extractRows(X, right_child_idxs), _extractRows1d(Y, right_child_idxs), depth + 1);
        return new TreeNode(-1., best_feature, best_cut, left_child_node, right_child_node, false);
    }
    // else not found
    auto value_node = _selectMostCommonValue(Y);
    return new TreeNode(value_node, -1, -1);
}

double DecisionTree::_find_proper_value(const Eigen::VectorXd &x, TreeNode *actual_node) {
    if (actual_node->isLeaf())
        return actual_node->value;
    if (x[actual_node->best_feature] <= actual_node->best_cut)
        return _find_proper_value(x, actual_node->left);
    return _find_proper_value(x, actual_node->right);
}

void DecisionTree::fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &Y) {
    vector<vector<double>> _X(X.rows(), vector<double>(X.cols()));
    vector<int> _Y(Y.rows());
    for (int i = 0; i < X.rows(); ++i)
        for (int j = 0; j < X.cols(); ++j)
            _X[i][j] = X(i, j);
    for (int i = 0; i < Y.rows(); ++i)
        _Y[i] = Y[i];
    this->root = _build_tree(_X, _Y);
}

vector<int> DecisionTree::predictRows(const Eigen::MatrixXd &X) {
    vector<int> prediction;
    for (int i = 0; i < X.rows(); ++i) {
        const Eigen::VectorXd &x = X.row(i);
        prediction.push_back(_find_proper_value(x, root));
    }
    return std::move(prediction);
}

int DecisionTree::predict(const Eigen::VectorXd &x) {
    return _find_proper_value(x, root);
}

