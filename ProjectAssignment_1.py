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
    print("Shape of features_matrix:", features_matrix.shape)  # Debugging
    mu = features_matrix.mean(1).reshape(features_matrix.shape[0],1)
    features_centered  = features_matrix - mu
    cov = 1/(features_matrix.shape[1]) * (features_centered @ features_centered.T)
    vals, vecs = numpy.linalg.eigh(cov)

    ## vecs contains in its columns the eigenvectors sorted corresponding to 
    ## the smallest eigenvalue to the largest
    ## we reverse the order:
    vecs_decreasing = vecs[:,::-1]
    P = vecs_decreasing[:,0:m]
    print(P)
    features_projected = numpy.dot(P.T,features_matrix)
    return features_projected

def Sw(features_matrix,labels):
    N = features_matrix.shape[1]
    Sw=0
    for label in labels:
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
    for label in labels:
        features_class = features_matrix[:, labels.flatten()==label]
        nc = features_class.shape[1]
        mu_c = features_class.mean(1).reshape(features_class.shape[0],1)
        Sb += nc*((mu_c-mu_dataset)@(mu_c-mu_dataset).T)
    
    Sb *= 1/N
    return Sb


if __name__ == '__main__':

    ## Data loading, visualization, and statistics --Assignment 1

    features, labels = load('trainData.txt')
    labels= labels.flatten()

    features_classT = features[:,labels.flatten()==1]
    features_classF = features[:,labels.flatten()==0]

    #statistics
    mu_classT = features_classT.mean(1).reshape(features_classT.shape[0],1)
    print(mu_classT)

    mu_classF = features_classF.mean(1).reshape(features_classF.shape[0],1)
    print(mu_classF)

    var_classT = features_classT.var(1)
    var_classF = features_classF.var(1)

    std_classT = features_classT.std(1)
    std_classF = features_classF.std(1)

    print(var_classT)
    print(var_classF)

    print(std_classT)
    print(std_classF)
 

    # PCA, LDA, & Classification --Assignment 2

    ## applying PCA
    PCAdata = PCA_projection(50,features)
    true_prints = PCAdata[:, labels.flatten()==0]
    false_prints = PCAdata[:, labels.flatten()==1]

    ## applying LDA
    mu_dataset = features.mean(1).reshape(features.shape[0],1)
    S_B = Sb(features,labels,mu_dataset)
    S_W = Sw(features,labels)

    s, U = scipy.linalg.eigh(S_B,S_W)
    W = U[:,::-1][:,0:1]  ## setting m to 1, 1-dimensional

    LDAdata = numpy.dot(W.T, features)
    true_prints = LDAdata[:, labels.flatten()==0]
    false_prints = LDAdata[:, labels.flatten()==1]

    print(true_prints.shape)
    print(false_prints.shape)

    plt.figure
    plt.hist(true_prints[0],bins=10, alpha=0.5, label= 'Genuine Fingerprints', color='green', density= True)
    plt.hist(false_prints[0],bins=10, alpha=0.5, label= 'Fake Fingerprints', color='red', density= True)
    plt.title("LDA projection")
    plt.legend()
    plt.show()

    ## LDA as a classifier
    print("Size of Labels: ", labels.shape)
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(features, labels)

    ## training LDA
    mu_TRdataset = DTR.mean(1).reshape(DTR.shape[0],1)

    S_B_LDA = Sb(DTR,LTR,mu_TRdataset)
    S_W_LDA = Sw(DTR,LTR)
    s1, U1 = scipy.linalg.eigh(S_B_LDA,S_W_LDA) #solving the general eigenvalue problem
    W_LDA = U1[:,::-1][:,0:1]  ## setting m to 1

    data_LDA_val_proj = numpy.dot(W_LDA.T, DVAL)
    data_LDA_train_proj = numpy.dot(W_LDA.T, DTR)
    true_prints = data_LDA_val_proj[:,LVAL.flatten()==0]
    false_prints = data_LDA_val_proj[:,LVAL.flatten()==1]
    mean_true = true_prints.mean(1).reshape(true_prints.shape[0],1)
    mean_false = false_prints.mean(1).reshape(false_prints.shape[0],1)
    print("true mean, false mean (after projection): ", mean_true, mean_false)
    

    threshold = (data_LDA_val_proj[:,LVAL==0].mean()+ data_LDA_train_proj[:,LTR==1].mean())/2.0

    plt.figure
    plt.hist(true_prints[0],bins=10, alpha=0.5, label= 'Genuine Fingerprints', color='green', density= True)
    plt.hist(false_prints[0],bins=10, alpha=0.5, label= 'Fake Fingerprints', color='red', density= True)
    plt.title("LDA Validation Set")
    plt.legend()
    plt.show()

    ## predicting labels

    ## tuning the threshold
    threshold = threshold*500

    PVAL = numpy.zeros(shape=LVAL.shape, dtype=numpy.int32)
    PVAL[data_LDA_val_proj[0]>=threshold] = 1
    PVAL[data_LDA_val_proj[0]<threshold] = 0


    counter= numpy.sum(LVAL != PVAL)
    print("Counter: ", counter)   ## counter = 185

    # reconstruction error
    print("Threshold: ", threshold)
    print("DVAL.shape[1]",DVAL.shape[1])

    
    error = 1/DVAL.shape[1] * (numpy.sum(numpy.square(data_LDA_val_proj-DVAL)))
    print("Error: ", error)

    # preprocessing PCA then LDA classification
    
