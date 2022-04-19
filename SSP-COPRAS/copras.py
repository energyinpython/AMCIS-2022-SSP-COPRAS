import numpy as np
from mcdm_method import MCDM_method

class COPRAS(MCDM_method):
    def __init__(self):
        """
        Create the COPRAS method object.
        """
        pass

    def __call__(self, matrix, weights, types, mad = False, s_coeff = 0):
        """
        Score alternatives provided in decision matrix `matrix` using criteria `weights` 
        and criteria `types`. `mad` parameter is set to False by default and it means that
        it is the calssical version of the COPRAS method. When `mad` is set to True,
        it is the SSP-COPRAS method considering reduced criteria compensation.

        Parameters
        ------------
            matrix : ndarray
                Decision matrix with m alternatives in rows and n criteria in columns.
            weights : ndarray
                Vector with criteria weights. The sum of weights must be equal to 1.
            types : ndarray
                Vector with criteria types. Profit criteria are represented by 1 and cost by -1.
            mad : bool
                Boolean variable. When `mad` is set to False, it means that it is the classical version
                of the COPRAS method without reduced criteria compensation. `mad` set to True means that
                it is the SSP-COPRAS method considering reduced criteria compensation
            s_coeff : ndarray
                Vector with values of sustainability coefficient for each criterion. It takes values
                from 0 to 1. 0 means full criteria compensation, and 1 represents a complete reduction
                of criteria compensation

        Returns
        ---------
        ndrarray
            Vector with preference values of each alternative. The best alternative has the highest 
            preference value, according to the COPRAS method.

        Examples
        ----------
        >>> copras = COPRAS()
        >>> pref = copras(matrix, weights, types)
        >>> rank = rank_preferences(pref, reverse = True)

        >>> s = 0.3
        >>> s_set = np.ones(matrix.shape[1]) * s
        >>> pref = copras(matrix, weights, types, mad = True, s_coeff = s_set)
        >>> rank = rank_preferences(pref, reverse = True)

        """
        COPRAS._verify_input_data(matrix, weights, types)
        return COPRAS._copras(matrix, weights, types, mad, s_coeff)

    @staticmethod
    def _copras(matrix, weights, types, mad, s_coeff):
        # Normalize matrix using the linear normalization
        norm_matrix = matrix/np.sum(matrix, axis = 0)
        # Create the weighted normalized decision matrix by multiplying normalized matrix by criteria weights
        # `mad` parameter set to True means that it is the SSP-COPRAS method
        if mad == True:
            std_val = norm_matrix - np.mean(norm_matrix, axis = 0)

            for j in range(norm_matrix.shape[1]):
                for i in range(norm_matrix.shape[0]):
                    if types[j] == 1:
                        if std_val[i, j] < 0:
                            std_val[i, j] = 0
                    elif types[j] == -1:
                        if std_val[i, j] > 0:
                            std_val[i, j] = 0

            std_val = std_val * s_coeff
            norm_matrix = norm_matrix - std_val
        d = norm_matrix * weights
        Sp = np.sum(d[:, types == 1], axis = 1)
        Sm = np.sum(d[:, types == -1], axis = 1)
        Q = Sp + ((np.sum(Sm))/(Sm * np.sum(1 / Sm)))
        # Q = Sp + ((np.min(Sm) * np.sum(Sm))/(Sm * np.sum(np.min(Sm) / Sm)))
        U = Q / np.max(Q)
        return U