import numpy
import matplotlib.pyplot as plt
import scipy.linalg
import sklearn

features = numpy.array([])
labels = numpy.array([])

def split_db_2to1(D, L, seed=0): 
    nTrain = int(D.shape[1]*2.0/3.0) 
    numpy.random.seed(seed) 
    idx = numpy.random.permutation(D.shape[1]) 
    idxTrain = idx[0:nTrain] 
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain] 
    DVAL = D[:, idxTest] 
    LTR = L[idxTrain] 
    LVAL = L[idxTest]
    return (DTR, LTR), (DVAL, LVAL)

def load(filename):
    list_of_arrays = []
    label_array = numpy.array([])

    with open(filename, 'r') as f:
       
        for line in f:
            a,b,c,d,e,f,label = line.split(',')
            list_of_arrays.append(numpy.array([a,b,c,d,e,f]).reshape(6,1))
            label_array = numpy.append(label_array, label.strip())

    feature_matrix = numpy.hstack(list_of_arrays)
    label_array = label_array.reshape(1,feature_matrix.shape[1])
  
    return feature_matrix.astype(float), label_array.astype(float)

def PCA_projection(m, features_matrix):
    mu = features_matrix.mean(1).reshape(features_matrix.shape[0],1)
    features_centered  = features_matrix - mu
    cov = 1/(features_matrix.shape[1]) * (features_centered @ features_centered.T)
    vals, vecs = numpy.linalg.eigh(cov)

    ## vecs contains in its columns the eigenvectors sorted corresponding to 
    ## the smallest eigenvalue to the largest
    ## we reverse the order:
    vecs_decreasing = vecs[:,::-1]
    P = vecs_decreasing[:,0:m]
    features_projected = numpy.dot(P.T,features_matrix)
    return features_projected, P

def Sw(features_matrix,labels):
    N = features_matrix.shape[1]
    Sw=0
    classes = set(labels)
    for label in classes:
        features = features_matrix[:,labels.flatten()==label]
        mu_c = features.mean(1).reshape(features.shape[0],1)
        nc = features.shape[1]  ##nb of samples of this class
        features_centered = features - mu_c
        cov = 1/nc * (features_centered @ features_centered.T)
        Sw += nc*cov
    
    Sw *= 1/N
    return Sw

def Sb(features_matrix,labels, mu_dataset):
    N = features_matrix.shape[1]
    Sb = 0
    classes = set(labels)
    for label in classes:
        features_class = features_matrix[:, labels.flatten()==label]
        nc = features_class.shape[1]
        mu_c = features_class.mean(1).reshape(features_class.shape[0],1)
        Sb += nc*((mu_c-mu_dataset)@(mu_c-mu_dataset).T)
    
    Sb *= 1/N
    return Sb

def computeStats(features_classT, featuresClassF):
    #statistics
    mu_classT = features_classT.mean(1).reshape(features_classT.shape[0],1)
    print("mean true class: ", mu_classT)

    mu_classF = features_classF.mean(1).reshape(features_classF.shape[0],1)
    print("mean false class: ", mu_classF)

    var_classT = features_classT.var(1)
    var_classF = features_classF.var(1)

    std_classT = features_classT.std(1)
    std_classF = features_classF.std(1)

    print("variance true class: ", var_classT)
    print("variance false class: ", var_classF)

    print("std deviation true class: ", std_classT)
    print("std deviation false class: ",std_classF)

def LDA_projection(features, labels):
    mu_dataset = features.mean(1).reshape(features.shape[0],1)
    S_B = Sb(features,labels,mu_dataset)
    S_W = Sw(features,labels)

    s, U = scipy.linalg.eigh(S_B,S_W)
    W = U[:,::-1][:,0:1]  ## setting m to 1, 1-dimensional


    LDAdata = numpy.dot(W.T, features)
    mean_true = LDAdata[0,labels==1].mean()
    mean_false = LDAdata[0,labels==0].mean()

    if(mean_true<mean_false):
        W=-W

    return LDAdata, W    

def LDA_Classifier(features,labels,pca_preprocessing,mPCA,plot):
    ## LDA as a classifier
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(features, labels)

    if pca_preprocessing:
        DTR, P = PCA_projection(mPCA,DTR)
        DVAL = numpy.dot(P.T,DVAL)

    ## training LDA
    LDA_TR, W = LDA_projection(DTR,LTR)
    LDA_VAL = numpy.dot(W.T, DVAL)

    if plot:
        train_true = LDA_TR[:,LTR.flatten()==1]
        train_false = LDA_TR[:,LTR.flatten()==0]
        val_true = LDA_VAL[:,LVAL.flatten()==1]
        val_false = LDA_VAL[:,LVAL.flatten()==0]

        plt.figure
        plt.hist(train_true[0],bins=10, alpha=0.5, label= 'Genuine Fingerprints', color='green', density= True)
        plt.hist(train_false[0],bins=10, alpha=0.5, label= 'Fake Fingerprints', color='red', density= True)
        plt.title("LDA Projection on Training Set")
        plt.legend()
        plt.show()

        plt.figure
        plt.hist(val_true[0],bins=10, alpha=0.5, label= 'Genuine Fingerprints', color='green', density= True)
        plt.hist(val_false[0],bins=10, alpha=0.5, label= 'Fake Fingerprints', color='red', density= True)
        plt.title("LDA Projection on Validation Set")
        plt.legend()
        plt.show()


    mean_true = LDA_TR[0,LTR==1].mean()
    mean_false = LDA_TR[0,LTR==0].mean()

    threshold = (mean_false+mean_true)/2
    PVAL = numpy.zeros(shape=LVAL.shape, dtype=numpy.int32)
    PVAL[LDA_VAL[0]>=threshold] = 1
    PVAL[LDA_VAL[0]<threshold] = 0

    wrong_samples = numpy.sum(LVAL!= PVAL)
    error = wrong_samples/LVAL.shape
    accuracy = 1-error
    correct_samples = LVAL.shape-wrong_samples

    return error, accuracy, correct_samples



if __name__ == '__main__':

    ## Data loading, visualization, and statistics --Assignment 1

    features, labels = load('trainData.txt')
    labels= labels.flatten()

    features_classT = features[:,labels.flatten()==1]
    features_classF = features[:,labels.flatten()==0]

    computeStats(features_classT,features_classF)
 
    # PCA, LDA, & Classification --Assignment 2

    ## applying PCA
    PCAdata,_ = PCA_projection(6,features)

    ## applying LDA
    LDAdata, W_LDA = LDA_projection(features,labels)

    ## LDA as a classifier
    '''
    error, accuracy, correct_samples = LDA_Classifier(features,labels,False,None,True)
    print("Error, Accuracy, Correct Samples: ",error,accuracy,correct_samples)
    '''

    # preprocessing with PCA then LDA classification
    error, accuracy, correct_samples = LDA_Classifier(features,labels,True,6,False)
    print("Error, Accuracy, Correct Samples: ",error,accuracy,correct_samples)
    