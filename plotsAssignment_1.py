import numpy
import matplotlib.pyplot as plt
import ProjectAssignment_1 as main

if __name__== '__main__':
    ### Analysing attributes: Histograms
        
    features, labels = main.load('trainData.txt')

    ## attribute 1
        
    attr1 = features[0,:]
    attr1_true = attr1[labels.flatten()==0]
    attr1_false= attr1[labels.flatten()==1]
    plt.figure()
    plt.hist(attr1_true, bins=7, alpha=0.5, label= 'Genuine Fingerprint', color='red', density= True)
    plt.hist(attr1_false, bins=7, alpha=0.5, label= 'Fake Fingerprint', color='green', density= True)
    plt.legend()
    plt.title('Attribute 1')
    plt.show()

    ## attribute 2

    attr2 = features[1,:]
    attr2_true = attr2[labels.flatten()==0]
    attr2_false= attr2[labels.flatten()==1]
    plt.figure()
    plt.hist(attr2_true, bins=7, alpha=0.5, label= 'Genuine Fingerprint', color='red', density= True)
    plt.hist(attr2_false, bins=7, alpha=0.5, label= 'Fake Fingerprint', color='green', density= True)
    plt.legend()
    plt.title('Attribute 2')
    plt.show()

    ## attribute 3

    attr3 = features[2,:]
    attr3_true = attr3[labels.flatten()==0]
    attr3_false= attr3[labels.flatten()==1]
    plt.figure()
    plt.hist(attr3_true, bins=7, alpha=0.5, label= 'Genuine Fingerprint', color='red', density= True)
    plt.hist(attr3_false, bins=7, alpha=0.5, label= 'Fake Fingerprint', color='green', density= True)
    plt.legend()
    plt.title('Attribute 3')
    plt.show()

    ## attribute 4

    attr4 = features[3,:]
    attr4_true = attr4[labels.flatten()==0]
    attr4_false= attr4[labels.flatten()==1]
    plt.figure()
    plt.hist(attr4_true, bins=7, alpha=0.5, label= 'Genuine Fingerprint', color='red', density= True)
    plt.hist(attr4_false, bins=7, alpha=0.5, label= 'Fake Fingerprint', color='green', density= True)
    plt.legend()
    plt.title('Attribute 4')
    plt.show()

    ## attribute 5

    attr5 = features[4,:]
    attr5_true = attr5[labels.flatten()==0]
    attr5_false= attr5[labels.flatten()==1]
    plt.figure()
    plt.hist(attr5_true, bins=7, alpha=0.5, label= 'Genuine Fingerprint', color='red', density= True)
    plt.hist(attr5_false, bins=7, alpha=0.5, label= 'Fake Fingerprint', color='green', density= True)
    plt.legend()
    plt.title('Attribute 5')
    plt.show()

    ## attribute 6

    attr6 = features[5,:]
    attr6_true = attr6[labels.flatten()==0]
    attr6_false= attr6[labels.flatten()==1]
    plt.figure()
    plt.hist(attr6_true, bins=7, alpha=0.5, label= 'Genuine Fingerprint', color='red', density= True)
    plt.hist(attr6_false, bins=7, alpha=0.5, label= 'Fake Fingerprint', color='green', density= True)
    plt.legend()
    plt.title('Attribute 6')
    plt.show()

    ### Analysing attributes: Scatter Plots

    ## attr1 vs attr2
    plt.figure()
    plt.scatter(attr1_true, attr2_true, label='Genuine')
    plt.scatter(attr1_false, attr2_false, label='Fake')
    plt.legend()
    plt.xlabel('Attribute 1')
    plt.ylabel('Attribute 2')
    plt.show()

    ## attr1 vs attr3
    plt.figure()
    plt.scatter(attr1_true, attr3_true, label='Genuine')
    plt.scatter(attr1_false, attr3_false, label='Fake')
    plt.legend()
    plt.xlabel('Attribute 1')
    plt.ylabel('Attribute 3')
    plt.show()

    ## attr1 vs attr4
    plt.figure()
    plt.scatter(attr1_true, attr4_true, label='Genuine')
    plt.scatter(attr1_false, attr4_false, label='Fake')
    plt.legend()
    plt.xlabel('Attribute 1')
    plt.ylabel('Attribute 4')
    plt.show()

    ## attr1 vs attr5
    plt.figure()
    plt.scatter(attr1_true, attr5_true, label='Genuine')
    plt.scatter(attr1_false, attr5_false, label='Fake')
    plt.legend()
    plt.xlabel('Attribute 1')
    plt.ylabel('Attribute 5')
    plt.show()

    ## attr1 vs attr6
    plt.figure()
    plt.scatter(attr1_true, attr6_true, label='Genuine')
    plt.scatter(attr1_false, attr6_false, label='Fake')
    plt.legend()
    plt.xlabel('Attribute 1')
    plt.ylabel('Attribute 6')
    plt.show()

    #################################

    ## attr2 vs attr3
    plt.figure()
    plt.scatter(attr2_true, attr3_true, label='Genuine')
    plt.scatter(attr2_false, attr3_false, label='Fake')
    plt.legend()
    plt.xlabel('Attribute 2')
    plt.ylabel('Attribute 3')
    plt.show()

    ## attr2 vs attr4
    plt.figure()
    plt.scatter(attr2_true, attr4_true, label='Genuine')
    plt.scatter(attr2_false, attr4_false, label='Fake')
    plt.legend()
    plt.xlabel('Attribute 2')
    plt.ylabel('Attribute 4')
    plt.show()

    ## attr2 vs attr5
    plt.figure()
    plt.scatter(attr2_true, attr5_true, label='Genuine')
    plt.scatter(attr2_false, attr5_false, label='Fake')
    plt.legend()
    plt.xlabel('Attribute 2')
    plt.ylabel('Attribute 5')
    plt.show()

    ## attr1 vs attr6
    plt.figure()
    plt.scatter(attr2_true, attr6_true, label='Genuine')
    plt.scatter(attr2_false, attr6_false, label='Fake')
    plt.legend()
    plt.xlabel('Attribute 2')
    plt.ylabel('Attribute 6')
    plt.show()





