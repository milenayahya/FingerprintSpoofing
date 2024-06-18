import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import ProjectAssignment_1 as main

if __name__ == '__main__':

    features, labels = main.load('trainData.txt')


    ## plotting the Histograms
    # Create a figure with 6 subplots arranged in 2 rows and 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Attributes
    attributes = [features[i, :] for i in range(6)]
    titles = [f'Feature {i+1}' for i in range(6)]
    color_genuine = '#4E8098'
    color_fake = '#A31621'

    for i, ax in enumerate(axes.flatten()):
        attr = attributes[i]
        attr_true = attr[labels.flatten() == 1]
        attr_false = attr[labels.flatten() == 0]
        
        ax.hist(attr_true, bins=20, alpha=0.5, label='Genuine Fingerprint', color=color_genuine, density=True)
        ax.hist(attr_false, bins=20, alpha=0.5, label='Fake Fingerprint', color=color_fake, density=True)
        ax.legend()
        ax.set_title(titles[i])

    plt.tight_layout()
    plt.show()

    ## plotting the 2D-Scatter plots

    data = pd.DataFrame(features.T, columns=[f'Feature {i+1}' for i in range(features.shape[0])])
    data['Class'] = np.where(labels.flatten() == 1, 'Genuine', 'Fake')

    # Create the PairGrid
    g = sns.PairGrid(data, hue='Class', palette={'Genuine': '#4E8098', 'Fake': '#A31621'}, diag_sharey=False)

    g.map_diag(sns.histplot, kde=False, alpha=0.5)
    g.map_offdiag(sns.scatterplot)

    g.add_legend()
    plt.show()
