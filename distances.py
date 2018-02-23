import _ERT


def ERT_tree_path_list(X, n_estimators, use_rank = False):

    if use_rank:
        X_use = _ERT.get_rank(X)
    else:
        X_use = X


    forest = _ERT.build_forest(X_use, n_estimators)
    trees = forest.estimators_

    tree_path = [_ERT.build_path_matrix(tree, X_use) for tree in trees]

    return tree_path


def ERT_similarity(idx, tree_path):
    """
    :param idx:  index of and object
    :param tree_path: tree_path list calculated in ERT_tree_path_list
    :return: the similarities between the object (with index idx) to all other objects
    """

    similarities_vec = _ERT.similarity_matrix_row(idx, tree_path)

    return similarities_vec





