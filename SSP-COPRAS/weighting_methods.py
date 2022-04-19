import numpy as np


# Gini coefficient-based weighting
def gini_weighting(matrix):
    """
    Calculate criteria weights using objective Gini coefficient-based weighting method.
    Parameters
    ----------
        matrix : ndarray
            Decision matrix with performance values of m alternatives and n criteria.
            
    Returns
    --------
        ndarray
            Vector of criteria weights.
    Examples
    ---------
    >>> weights = gini_weighting(matrix)
    """
    m, n = np.shape(matrix)
    G = np.zeros(n)
    # Calculate the Gini coefficient for decision matrix `matrix`
    # iteration over criteria j = 1, 2, ..., n
    for j in range(n):
        Yi = 0
        # iteration over alternatives i = 1, 2, ..., m
        if np.mean(matrix[:, j]):
            for i in range(m):
                Yi += np.sum(np.abs(matrix[i, j] - matrix[:, j]) / (2 * m**2 * (np.sum(matrix[:, j]) / m)))
        else:
            for i in range(m):
                Yi += np.sum(np.abs(matrix[i, j] - matrix[:, j]) / (m**2 - m))

        G[j] = Yi
    # calculate and return the criteria weights by dividing the vector of Gini coefficients by their sum
    w = G / np.sum(G)
    return w
