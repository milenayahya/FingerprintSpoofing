import numpy
import matplotlib.pyplot as plt
import ProjectAssignment_1 as main

if __name__== '__main__':
    
    features, labels = main.load('trainData.txt')
    PCAdata = main.PCA_projection(50, features)
    print(PCAdata)

    
    true_prints = PCAdata[:, labels.flatten()==0]
    false_prints = PCAdata[:, labels.flatten()==1]

    ## histograms
    plt.figure()
    plt.hist(true_prints[0,:], bins=7, alpha=0.5, label= 'Genuine Fingerprint', color='green', density= True)
    plt.hist(false_prints[0,:], bins=7, alpha=0.5, label= 'Fake Fingerprint', color='red', density= True)
    plt.legend()
    plt.title('Principal Component 1')
    plt.show()

    plt.figure()
    plt.hist(true_prints[1,:], bins=7, alpha=0.5, label= 'Genuine Fingerprint', color='green', density= True)
    plt.hist(false_prints[1,:], bins=7, alpha=0.5, label= 'Fake Fingerprint', color='red', density= True)
    plt.legend()
    plt.title('Principal Component 2')
    plt.show()

    plt.figure()
    plt.hist(true_prints[2,:], bins=7, alpha=0.5, label= 'Genuine Fingerprint', color='green', density= True)
    plt.hist(false_prints[2,:], bins=7, alpha=0.5, label= 'Fake Fingerprint', color='red', density= True)
    plt.legend()
    plt.title('Principal Component 3')
    plt.show()

    plt.figure()
    plt.hist(true_prints[3,:], bins=7, alpha=0.5, label= 'Genuine Fingerprint', color='green', density= True)
    plt.hist(false_prints[3,:], bins=7, alpha=0.5, label= 'Fake Fingerprint', color='red', density= True)
    plt.legend()
    plt.title('Principal Component 4')
    plt.show()

    plt.figure()
    plt.hist(true_prints[4,:], bins=7, alpha=0.5, label= 'Genuine Fingerprint', color='green', density= True)
    plt.hist(false_prints[4,:], bins=7, alpha=0.5, label= 'Fake Fingerprint', color='red', density= True)
    plt.legend()
    plt.title('Principal Component 5')
    plt.show()

    plt.figure()
    plt.hist(true_prints[5,:], bins=7, alpha=0.5, label= 'Genuine Fingerprint', color='green', density= True)
    plt.hist(false_prints[5,:], bins=7, alpha=0.5, label= 'Fake Fingerprint', color='red', density= True)
    plt.legend()
    plt.title('Principal Component 6')
    plt.show()

    ## first two directions
    true_1 = true_prints[:2,:]
    false_1 = false_prints[:2,:]
    plt.figure
    plt.scatter(true_1[0],true_1[1], label= 'Genuine Fingerprints', color='green')
    plt.scatter(false_1[0],false_1[1], label= 'Fake Fingerprints', color='red')
    plt.title('2D PCA Projection of Data')
    plt.legend()
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.show()

    ## second two directions
    true_2 = true_prints[2:4,:]
    false_2 = false_prints[2:4,:]
    plt.figure
    plt.scatter(true_2[0],true_2[1], label= 'Genuine Fingerprints', color='green')
    plt.scatter(false_2[0],false_2[1], label= 'Fake Fingerprints', color='red')
    plt.title('2D PCA Projection of Data')
    plt.legend()
    plt.xlabel('Principal Component 3')
    plt.ylabel('Principal Component 4')
    plt.grid(True)
    plt.show()

    
    ## third two directions
    true_3 = true_prints[4:,:]
    false_3 = false_prints[4:,:]
    plt.figure
    plt.scatter(true_3[0],true_3[1], label= 'Genuine Fingerprints', color='green')
    plt.scatter(false_3[0],false_3[1], label= 'Fake Fingerprints', color='red')
    plt.title('2D PCA Projection of Data')
    plt.legend()
    plt.xlabel('Principal Component 5')
    plt.ylabel('Principal Component 6')
    plt.grid(True)
    plt.show()