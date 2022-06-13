#ifndef HOW_TO_PASS_ART_CLASSES_TREENODE_H
#define HOW_TO_PASS_ART_CLASSES_TREENODE_H


class TreeNode {
public:
    double value, best_cut;
    int best_feature;
    TreeNode *left, *right;
    bool is_leaf;
    TreeNode(double _value, int _best_feature, double _best_cut, TreeNode *_left = nullptr, TreeNode *_right = nullptr, bool _is_leaf = true):
        value(_value), best_feature(_best_feature), best_cut(_best_cut), left(_left), right(_right), is_leaf(_is_leaf)
        {}
    ~TreeNode() {
        if (left != nullptr)
            delete left;
        if (right != nullptr)
            delete right;
    }
    bool isLeaf() {
        return is_leaf;
    }
};


#endif //HOW_TO_PASS_ART_CLASSES_TREENODE_H
