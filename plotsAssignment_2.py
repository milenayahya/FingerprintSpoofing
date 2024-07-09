import numpy
import matplotlib.pyplot as plt
import projectScript as main

if __name__== '__main__':

    ## PCA plots
    
    features, labels = main.load('trainData.txt')
    PCAdata, _ = main.PCA_projection(6, features)
    
    true_prints = PCAdata[:, labels.flatten() == 1]
    false_prints = PCAdata[:, labels.flatten() == 0]

    ## Plotting the Histograms
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Principal Components Titles
    titles = [f'Principal Component {i+1}' for i in range(6)]
    color_genuine = '#4E8098'
    color_fake = '#A31621'

    for i, ax in enumerate(axes.flatten()):
        
        ax.hist(true_prints[i,:], bins=20, alpha=0.5, label='Genuine Fingerprint', color=color_genuine, density=True)
        ax.hist(false_prints[i,:], bins=20, alpha=0.5, label='Fake Fingerprint', color=color_fake, density=True)
        ax.legend()
        ax.set_title(titles[i])

    plt.tight_layout()
    plt.show()


    ## Scatter plots for the first three pairs of principal components in one figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    pairs = [(0, 1), (2, 3), (4, 5)]
    component_labels = [
        ('Principal Component 1', 'Principal Component 2'),
        ('Principal Component 3', 'Principal Component 4'),
        ('Principal Component 5', 'Principal Component 6')
    ]

    for ax, (i, j), (xlabel, ylabel) in zip(axes, pairs, component_labels):
        true_pair = true_prints[[i, j], :]
        false_pair = false_prints[[i, j], :]
        ax.scatter(true_pair[0], true_pair[1], label='Genuine Fingerprints', color=color_genuine)
        ax.scatter(false_pair[0], false_pair[1], label='Fake Fingerprints', color=color_fake)
        ax.set_title(f'2D PCA Projection of Data\n({xlabel} & {ylabel})')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()

    ## LDA plots
    labels= labels.flatten()
    LDAdata, W_LDA = main.LDA_projection(features,labels)
    true_prints = LDAdata[:, labels.flatten()==1]
    false_prints = LDAdata[:, labels.flatten()==0]

    plt.figure
    plt.hist(true_prints[0],bins=20, alpha=0.5, label= 'Genuine Fingerprints', color=color_genuine, density= True)
    plt.hist(false_prints[0],bins=20, alpha=0.5, label= 'Fake Fingerprints', color=color_fake, density= True)
    plt.title("LDA projection")
    plt.legend()
    plt.show()
