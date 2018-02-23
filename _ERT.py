import numpy
from numba import jit
from sklearn.ensemble import IsolationForest

global PATH_MATRIX_EMPTY
PATH_MATRIX_EMPTY = 2**15-1



def get_rank(X_values):
    """
    Replace feature values with rank values
    For each feature, rank the objects according to their feature value.
    These rank values are the new features.

    :param X_values: data matrix
    :return: rank value matrix
    """
    nof_objects = X_values.shape[0]
    if len(X_values.shape) > 1:
        nof_features = X_values.shape[1]
    else:
        nof_features = 1

    X_loc = numpy.arange(nof_objects)
    X_rank = numpy.zeros([nof_objects, nof_features], dtype=int)

    X_argsort = X_values.argsort(axis=0)
    for i in range(nof_features):
        X_rank[X_argsort[:, i], i] = X_loc

    return X_rank


def build_forest(X, n_estimators):
    """
    :param X: data
    :param n_estimators: number of trees
    :return: Extremely randomized trees forest
    """
    forest = IsolationForest(n_jobs=-1, n_estimators=n_estimators)
    forest.fit(X)
    return forest

def build_path_matrix(tree, X):
    """
    :param tree: a decision tree
    :param X: data (matching to train data of the decision tree)
    :return: a matrix containing the path each object in X goes through in the tree
    """
    """
    This function receives a single tree and X (the data) (where the tree can be applied on X).
    For each tree it builds a matrix in which rows are objects, columns are levels in the tree. 
    Matrix element is which node this objects was on in the given level (level = column).
    """
    nof_objects = X.shape[0]
    all_path = tree.decision_path(X)
    #all_path is basically what we want, now we just put it in a form of a matrix

    """
    Number of nodes object went through in the tree
    """
    nof_nodes = numpy.zeros(nof_objects, dtype=numpy.dtype('i4'))
    for i in range(nof_objects):
        nof_nodes[i] = all_path.getrow(i).count_nonzero()

    max_nof_nodes = nof_nodes.max()


    path_matrix = numpy.ones(([nof_objects, max_nof_nodes]), dtype=numpy.dtype('i4')) * PATH_MATRIX_EMPTY
    for i in range(nof_objects):
        path_matrix[i, :nof_nodes[i]] = all_path.getrow(i).indices

    return path_matrix








@jit
def similarity_matrix_row(i, path_matrix_list):
    """
    In this implementation of the distance matrix,
    the number of node two objects went trough together is the similarity
    """
    similarity_vec = 0
    nof_objects = 0
    is_first_iter = True
    n_trees = len(path_matrix_list)

    for path_matrix in path_matrix_list:

        if is_first_iter:
            # init distance vector on first iteration of the loop
            path_matrix = path_matrix_list[0]
            nof_objects = path_matrix.shape[0]
            similarity_vec = numpy.zeros(nof_objects, dtype = numpy.dtype('f4'))
            is_first_iter = False

        i_final_node = numpy.searchsorted(path_matrix[i], v=PATH_MATRIX_EMPTY, side='left')

        for j in range(i,nof_objects):
            similarity = 0
            for l in range(i_final_node-1,-1,-1):
                if (path_matrix[i,l] == path_matrix[j,l]):
                    similarity = l
                    break
            similarity_vec[j] = similarity_vec[j] + similarity
    similarity_vec = similarity_vec / float(n_trees)


    return similarity_vec