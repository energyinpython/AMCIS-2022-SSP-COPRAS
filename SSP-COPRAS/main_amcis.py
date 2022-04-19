import numpy as np
import pandas as pd
import copy
import os
import matplotlib.pyplot as plt

from visualizations import plot_sustainability
from weighting_methods import gini_weighting

from rank_preferences import rank_preferences
from copras import COPRAS

def main():
    # Criteria indexes of each main dimension.
    modules_indexes = [
        [0, 1, 2, 3, 4, 5, 6],
        [7, 8, 9, 10, 11, 12, 13],
        [14, 15, 16],
        [17, 18, 19],
        [20, 21, 22, 23, 24]
    ]

    G = [r'$G_{' + str(i + 1) + '}$' for i in range(len(modules_indexes))]
    # Name of folder with input data
    folder_name = './dataset'
    # Name of the file containing the dataset
    file_name = 'data_2019.csv'
    path_data = os.path.join(folder_name, file_name)
    data = pd.read_csv(path_data, index_col = 'Country')

    df_data = data.iloc[:len(data) - 1, :]
    df_types = data.iloc[len(data) - 1, :]
    types = df_types.to_numpy()

    list_alt_names = [r'$A_{' + str(i + 1) + '}$' for i in range(0, len(df_data))]
    results = pd.DataFrame(index = list_alt_names)
    df_compared = pd.DataFrame(index = list_alt_names)

    matrix = df_data.to_numpy()

    weights_type = 'gini'
    # Calculate criteria weights using an objective weighting method called Gini coefficient-based weighting method.
    weights = gini_weighting(matrix)

    # Save calculated criteria weights to the CSV file.
    crit_list = [r'$C_{' + str(i + 1) + '}$' for i in range(0, df_data.shape[1])]
    df_weights = pd.DataFrame(weights.reshape(1, -1), index = ['Weights'], columns = crit_list)
    df_weights.to_csv('results/weights_' + weights_type + '.csv')

    # Initialize the COPRAS method object.
    copras = COPRAS()
    # Calculate COPRAS preference values of alternatives.
    pref = copras(matrix, weights, types)
    # Rank alternatives according to COPRAS preference values in descending order. The option with the highest COPRAS preference value is the best one.
    rank = rank_preferences(pref, reverse = True)

    # Save results including preference values and ranking in DataFrame and CSV file.
    results['Utility'] = pref
    results['Rank'] = rank
    df_compared['Full compensation'] = rank
    print(results)
    results.to_csv('results/copras_results.csv')


    plt.style.use('seaborn')
    #
    # Apply the SSP-AHP method.
    # Changes in the coefficient s in all criteria dimensions are applied simultaneously
    # The value 0 of the s coefficient corresponds to the application of the classical COPRAS method.
    # Create the DataFrame for rankings for different sustainability coefficient values.
    df_sust = pd.DataFrame(index = list_alt_names)
    # Create the DataFrame for COPRAS preference values for different sustainability coefficient values.
    df_sust_pref = copy.deepcopy(df_sust)
    sust_coeffs = np.arange(0, 1.05, 0.05)
    # Iterate by each value of the sustainability coefficient.
    for s in sust_coeffs:
        s_set = np.ones(matrix.shape[1]) * s
        # `mad` parameter set to True means that it is the SSP-COPRAS method used with the s coefficient
        # `mad` is set to False by default for the calssical version of the COPRAS method
        # Calculate the SSP-COPRAS preference values for a given sustainability coefficient.
        pref = copras(matrix, weights, types, mad = True, s_coeff = s_set)
        df_sust_pref[str(s)] = pref
        # Determine the SSP-COPRAS ranking for a given sustainability coefficient.
        rank = rank_preferences(pref, reverse = True)
        df_sust[str(s)] = rank
    # Save results from DataFrames to CSV files.
    df_sust_pref.to_csv('results/sust_utility_vals_' + weights_type + '.csv')
    df_sust.to_csv('results/sust_rank_' + weights_type + '.csv')
    # Display chart of rankings obtained for each sustainability coefficient value.
    plot_sustainability(sust_coeffs, df_sust, weights_type)

    # Save rankings for full criteria compensation and fully reduced criteria compensation in CSV.
    df_compared['Reduced compensation'] = rank
    df_compared.to_csv('results/compared.csv')

if __name__ == '__main__':
    main()