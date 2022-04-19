import matplotlib.pyplot as plt


def plot_sustainability(vec, data_sust, weights_type = '', no = '', title = ''):
    """
    This function is used to plot alternatives rankings considering their changing
    due to changing sustainability coefficient.

    Parameters
    ------------
        vec : ndarray
            Vector with values of sustainability coefficient for each criterion.
        data_sust : dataframe
            DataFrame including particular rankings in columns for each value of 
            sustainability coeffcient
        weights_type : str
            Name of criteria weighting method
        no : str
            Number of chart (optional, if you want to plot multiple figures)
        title : str
            Tile of chart (optional)

    Examples
    ---------
    >>> plot_sustainability(sust_coeffs, df_sust, weights_type)
    """
    color = []
    for i in range(8):
        color.append(plt.cm.Set1(i))

    for i in range(8):
        color.append(plt.cm.Dark2(i))

    vec = vec * 100
    plt.figure(figsize = (7, 4))
    for j in range(data_sust.shape[0]):
        
        plt.plot(vec, data_sust.iloc[j, :], color = color[j], linewidth = 2)
        ax = plt.gca()
        y_min, y_max = ax.get_ylim()
        x_min, x_max = ax.get_xlim()
        plt.annotate(data_sust.index[j], (x_max + 0.05, data_sust.iloc[j, -1]),
                        fontsize = 14, style='italic',
                        horizontalalignment='left')

    plt.xlabel(r'$S$' + ' coefficient [%]', fontsize = 14)
    plt.ylabel("Rank", fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.xticks(ticks=vec, fontsize = 14)
    plt.title(title)
    plt.grid(True, linewidth = 1)
    plt.tight_layout()
    plt.savefig('./results_png/' + no + 'sustainability_' + weights_type + '.png')
    plt.savefig('./results/' + no + 'sustainability_' + weights_type + '.pdf')
    plt.show()