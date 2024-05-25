import numpy
import matplotlib.pyplot as plt
import ProjectAssignment_1 as main

if __name__== '__main__':
    ### Analysing attributes: Histograms
        
    features, labels = main.load('trainData.txt')

    ## attribute 1
        
    attr1 = features[0,:]
    attr1_true = attr1[labels.flatten()==1]
    attr1_false= attr1[labels.flatten()==0]
    plt.figure()
    plt.hist(attr1_true, bins=7, alpha=0.5, label= 'Genuine Fingerprint', color='green', density= True)
    plt.hist(attr1_false, bins=7, alpha=0.5, label= 'Fake Fingerprint', color='red', density= True)
    plt.legend()
    plt.title('Attribute 1')
    plt.show()

    ## attribute 2

    attr2 = features[1,:]
    attr2_true = attr2[labels.flatten()==1]
    attr2_false= attr2[labels.flatten()==0]
    plt.figure()
    plt.hist(attr2_true, bins=7, alpha=0.5, label= 'Genuine Fingerprint', color='green', density= True)
    plt.hist(attr2_false, bins=7, alpha=0.5, label= 'Fake Fingerprint', color='red', density= True)
    plt.legend()
    plt.title('Attribute 2')
    plt.show()

    ## scatter attr1 and attr2

    plt.figure()
    plt.scatter(attr1_true, attr2_true, color='green', label= 'Genuine Fingerprint')
    plt.scatter(attr1_false, attr2_false, color='red', label= 'Fake Fingerprint')
    plt.xlabel('Attribute 1')
    plt.ylabel('Attribute 2')
    plt.legend()
    plt.show()

    mu_T1= attr1_true.mean()
    mu_T2 = attr2_true.mean()
    mu_F1 = attr1_false.mean()
    mu_F2 = attr2_false.mean()

    ## attribute 3

    attr3 = features[2,:]
    attr3_true = attr3[labels.flatten()==1]
    attr3_false= attr3[labels.flatten()==0]
    plt.figure()
    plt.hist(attr3_true, bins=7, alpha=0.5, label= 'Genuine Fingerprint', color='green', density= True)
    plt.hist(attr3_false, bins=7, alpha=0.5, label= 'Fake Fingerprint', color='red', density= True)
    plt.legend()
    plt.title('Attribute 3')
    plt.show()

    ## attribute 4

    attr4 = features[3,:]
    attr4_true = attr4[labels.flatten()==1]
    attr4_false= attr4[labels.flatten()==0]
    plt.figure()
    plt.hist(attr4_true, bins=7, alpha=0.5, label= 'Genuine Fingerprint', color='green', density= True)
    plt.hist(attr4_false, bins=7, alpha=0.5, label= 'Fake Fingerprint', color='red', density= True)
    plt.legend()
    plt.title('Attribute 4')
    plt.show()

    ## scatter attr3 and attr4

    plt.figure()
    plt.scatter(attr3_true, attr4_true, color='green', label= 'Genuine Fingerprint')
    plt.scatter(attr3_false, attr4_false, color='red', label= 'Fake Fingerprint')
    plt.xlabel('Attribute 3')
    plt.ylabel('Attribute 4')
    plt.legend()
    plt.show()


    ## attribute 5

    attr5 = features[4,:]
    attr5_true = attr5[labels.flatten()==1]
    attr5_false= attr5[labels.flatten()==0]
    plt.figure()
    plt.hist(attr5_true, bins=7, alpha=0.5, label= 'Genuine Fingerprint', color='green', density= True)
    plt.hist(attr5_false, bins=7, alpha=0.5, label= 'Fake Fingerprint', color='red', density= True)
    plt.legend()
    plt.title('Attribute 5')
    plt.show()

    ## attribute 6

    attr6 = features[5,:]
    attr6_true = attr6[labels.flatten()==1]
    attr6_false= attr6[labels.flatten()==0]
    plt.figure()
    plt.hist(attr6_true, bins=7, alpha=0.5, label= 'Genuine Fingerprint', color='green', density= True)
    plt.hist(attr6_false, bins=7, alpha=0.5, label= 'Fake Fingerprint', color='red', density= True)
    plt.legend()
    plt.title('Attribute 6')
    plt.show()

    ## scatter attr5 and attr6

    plt.figure()
    plt.scatter(attr5_true, attr6_true, color='green', label= 'Genuine Fingerprint')
    plt.scatter(attr5_false, attr6_false, color='red', label= 'Fake Fingerprint')
    plt.xlabel('Attribute 5')
    plt.ylabel('Attribute 6')
    plt.legend()
    plt.show()

  
    ## statistics based on class + attribute

    
    mu_T1= attr1_true.mean()
    mu_T2 = attr2_true.mean()
    mu_F1 = attr1_false.mean()
    mu_F2 = attr2_false.mean()

    mu_T3= attr3_true.mean()
    mu_T4= attr4_true.mean()
    mu_F3 = attr3_false.mean()
    mu_F4 = attr4_false.mean()

    mu_T5= attr5_true.mean()
    mu_T6= attr6_true.mean()
    mu_F5 = attr5_false.mean()
    mu_F6 = attr6_false.mean()

    varT1 = attr1_true.var()
    varT2 = attr2_true.var()
    varT3 = attr3_true.var()
    varT4 = attr4_true.var()
    varT5 = attr5_true.var()
    varT6 = attr6_true.var()

    varF1 = attr1_false.var()
    varF2 = attr2_false.var()
    varF3 = attr3_false.var()
    varF4 = attr4_false.var()
    varF5 = attr5_false.var()
    varF6 = attr6_false.var()

    print("Mean of true fingerprints, attribute 1: ", mu_T1)
    print("Mean of true fingerprints, attribute 2: ", mu_T2)
    print("Mean of true fingerprints, attribute 3: ", mu_T3)
    print("Mean of true fingerprints, attribute 4: ", mu_T4)
    print("Mean of true fingerprints, attribute 5: ", mu_T5)
    print("Mean of true fingerprints, attribute 6: ", mu_T6)

    print("Mean of false fingerprints, attribute 1: ", mu_F1)
    print("Mean of false fingerprints, attribute 2: ", mu_F2)
    print("Mean of false fingerprints, attribute 3: ", mu_F3)
    print("Mean of false fingerprints, attribute 4: ", mu_F4)
    print("Mean of false fingerprints, attribute 5: ", mu_F5)
    print("Mean of false fingerprints, attribute 6: ", mu_F6)

    print("Variance of true fingerprints, attribute 1: ", varT1)
    print("Variance of true fingerprints, attribute 2: ", varT2)
    print("Variance of true fingerprints, attribute 3: ", varT3)
    print("Variance of true fingerprints, attribute 4: ", varT4)
    print("Variance of true fingerprints, attribute 5: ", varT5)
    print("Variance of true fingerprints, attribute 6: ", varT6)

    print("Variance of false fingerprints, attribute 1: ", varF1)
    print("Variance of false fingerprints, attribute 2: ", varF2)
    print("Variance of false fingerprints, attribute 3: ", varF3)
    print("Variance of false fingerprints, attribute 4: ", varF4)
    print("Variance of false fingerprints, attribute 5: ", varF5)
    print("Variance of false fingerprints, attribute 6: ", varF6)

    
    
