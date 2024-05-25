import numpy
import matplotlib.pyplot as plt
import scipy.linalg
import sklearn

features = numpy.array([])
labels = numpy.array([])


def vcol(vector):
    return vector.reshape(-1,1)

def vrow(vector):
    return vector.reshape(1,-1)

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

def LDA_Classifier(DTR,LTR,DVAL,LVAL,pca_preprocessing,mPCA,plot):
    ## LDA as a classifier
    
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

def logpdf_GAU_ND(X,mu,C):

    M = X.shape[0]
    N= X.shape[1]
    ND = []
    for i in range(N):
        x = X[:,i].reshape(X.shape[0],1)
        _, log_detC = numpy.linalg.slogdet(C)

        term1= -M/2*numpy.log(2*numpy.pi)
        term2 = -1/2*log_detC
        term3 = -1/2*(x-mu).T@numpy.linalg.inv(C)@(x-mu)
        result = term1+term2+term3
        ND.append(result)
    
    #flatten the list: list of arrays -> list of numbers
    NDD = [n[0][0] for n in ND]
    return NDD

def ML(X):
    mean = X.mean(1).reshape(X.shape[0],1)
    X_centered = X-mean
    cov = 1/(X.shape[1])*(X_centered@ X_centered.T)
    return mean, cov

def loglikelihood(XND, m_ML, C_ML):
    
    NDD= logpdf_GAU_ND(XND,m_ML,C_ML)
    l=sum(NDD)
    return l

def MVG(DTR, LTR, DVAL, prior):
    DF= DTR[:, LTR.flatten()==0]
    DT= DTR[:, LTR.flatten()==1]

    mF, cF = ML(DF)
    mT, cT = ML(DT)

    likelihoodF = numpy.exp(logpdf_GAU_ND(DVAL,mF,cF)).reshape(1,DVAL.shape[1])
    likelihoodT = numpy.exp(logpdf_GAU_ND(DVAL,mT,cT)).reshape(1,DVAL.shape[1])
    S= numpy.vstack((likelihoodF,likelihoodT))
    SJoint= prior*S
    SMarginal = vrow(SJoint.sum(0))
    SPost = SJoint/SMarginal

    return SPost, likelihoodF, likelihoodT

def evaluate_model(SPost, LTE, predicted_labels=None, llr=False):

    if llr==False:
        predicted_labels = numpy.argmax(SPost,axis=0)

    #else we have the predicted_labels from llr
    accuracy = numpy.sum(predicted_labels==LTE)/LTE.shape[0]
    error= 1- accuracy
    
    return accuracy,error

def NaiveBayes(DTR,LTR,DVAL,prior):

    DF= DTR[:, LTR.flatten()==0]
    DT= DTR[:, LTR.flatten()==1]

    mF, cF = ML(DF)
    mT, cT = ML(DT)

    n= cF.shape[0]
    I = numpy.eye(n)
    cFNB = numpy.multiply(cF,I)
    cTNB = numpy.multiply(cT,I)

    likelihoodF = numpy.exp(logpdf_GAU_ND(DVAL,mF,cFNB)).reshape(1,DVAL.shape[1])
    likelihoodT = numpy.exp(logpdf_GAU_ND(DVAL,mT,cTNB)).reshape(1,DVAL.shape[1])

    S= numpy.vstack((likelihoodF,likelihoodT))
    SJoint= prior*S
    SMarginal = vrow(SJoint.sum(0))
    SPost = SJoint/SMarginal

    return SPost, likelihoodF, likelihoodT

def TiedCov(DTR,LTR,DVAL,prior):
    SW = Sw(DTR,LTR)
    
    DF= DTR[:, LTR.flatten()==0]
    DT= DTR[:, LTR.flatten()==1]

    mF, _ = ML(DF)
    mT, _ = ML(DT)
    likelihoodF = numpy.exp(logpdf_GAU_ND(DVAL,mF,SW)).reshape(1,DVAL.shape[1])
    likelihoodT = numpy.exp(logpdf_GAU_ND(DVAL,mT,SW)).reshape(1,DVAL.shape[1])

    S= numpy.vstack((likelihoodF,likelihoodT))
    SJoint= prior*S
    SMarginal = vrow(SJoint.sum(0))
    SPost = SJoint/SMarginal

    return SPost, likelihoodF, likelihoodT

def llr(likelihoodF, likelihoodT, threshold):
    score = numpy.log(likelihoodT) - numpy.log(likelihoodF)
    predicted_labels =numpy.where(score>=threshold, 1, 0)
    return predicted_labels, score

def effective_prior(pi,Cfn,Cfp):
    pi_tilde = (pi*Cfn)/((pi*Cfn)*((1-pi)*Cfp))
    return pi_tilde

def Bayes_Decision(llr,pi, Cfn, Cfp):
    t = -numpy.log((pi*Cfn)/((1-pi)*Cfp))
    c= numpy.where(llr<=t,0,1)
    return c

def min_cost(llr,thresholds,labels,pi,Cfn,Cfp):
    res =[]
    Pfn =[]
    Pfp =[]
    for t in thresholds:
        c= numpy.where(llr<=t,0,1)
        conf_mat = compute_conf_matrix(c,labels)
        dcf, dcf_norm, pfn,pfp = binary_dcf(c,labels,pi,Cfn,Cfp)
       
        res.append(dcf_norm)
        Pfn.append(pfn)
        Pfp.append(pfp)
           
    minDCF = min(res)

    return minDCF,Pfn,Pfp

def compute_conf_matrix(predictions,labels):
    i=0
    nb_classes = len(numpy.unique(labels))
    conf_matrix= numpy.zeros((nb_classes,nb_classes))
    for value in numpy.unique(labels):
        prediction_per_class= predictions[labels==value]
        prediction = numpy.zeros((1,nb_classes))
        for class_k in range(nb_classes):
            prediction[0,class_k] = numpy.sum(prediction_per_class==class_k)
        conf_matrix[:,i] = vcol(prediction)[:,0]
        i+=1
    return conf_matrix

def binary_dcf(predictions, labels, pi,Cfn,Cfp):
    conf_matrix= compute_conf_matrix(predictions,labels)
    Pfp = conf_matrix[1,0]/(conf_matrix[1,0]+conf_matrix[0,0])
    Pfn = conf_matrix[0,1]/(conf_matrix[0,1]+conf_matrix[1,1])
    dcf = pi*Cfn*Pfn + ((1-pi)*Cfp*Pfp)
    B_dummy = min(pi*Cfn, (1-pi)*Cfp)
    dcf_norm = dcf/B_dummy
    return dcf,dcf_norm,Pfn,Pfp

def bayes_error_plot(effPriorLogOdds, llr_MVG, llr_TC, llr_NB, labels):
    DCF_MVG= numpy.array([])
    DCF_MIN_MVG= numpy.array([])

    DCF_TC= numpy.array([])
    DCF_MIN_TC= numpy.array([])

    DCF_NB= numpy.array([])
    DCF_MIN_NB= numpy.array([])

    for p in effPriorLogOdds:
        pi_tilde = 1/ (1 +numpy.exp(-p))
        #compute min_dcf
        thresholds_MVG = numpy.sort(llr_MVG)
        minDCF_MVG,Pfn,Pfp= min_cost(vcol(llr_MVG),vcol(thresholds_MVG),labels,pi_tilde,1,1)

        thresholds_TC = numpy.sort(llr_TC)
        minDCF_TC,Pfn,Pfp= min_cost(vcol(llr_TC),vcol(thresholds_TC),labels,pi_tilde,1,1)

        thresholds_NB = numpy.sort(llr_NB)
        minDCF_NB,Pfn,Pfp= min_cost(vcol(llr_NB),vcol(thresholds_NB),labels,pi_tilde,1,1)

        #compute DCF
        predictions_MVG = Bayes_Decision(llr_MVG,pi_tilde,1,1)
        dcf, dcf_norm_MVG,_,_ = binary_dcf(vcol(predictions_MVG),labels,pi_tilde,1,1)

        predictions_TC = Bayes_Decision(llr_TC,pi_tilde,1,1)
        dcf, dcf_norm_TC,_,_ = binary_dcf(vcol(predictions_TC),labels,pi_tilde,1,1)

        predictions_NB = Bayes_Decision(llr_NB,pi_tilde,1,1)
        dcf, dcf_norm_NB,_,_ = binary_dcf(vcol(predictions_NB),labels,pi_tilde,1,1)

        if DCF_MVG is None:
            DCF_MVG = dcf_norm_MVG
            DCF_MIN_MVG = minDCF_MVG

            DCF_TC = dcf_norm_TC
            DCF_MIN_TC = minDCF_TC

            DCF_NB = dcf_norm_NB
            DCF_MIN_NB = minDCF_NB
        else:
            DCF_MVG = numpy.append(DCF_MVG,dcf_norm_MVG)
            DCF_MIN_MVG = numpy.append(DCF_MIN_MVG,minDCF_MVG)

            DCF_TC = numpy.append(DCF_TC,dcf_norm_TC)
            DCF_MIN_TC = numpy.append(DCF_MIN_TC,minDCF_TC)
           
            DCF_NB = numpy.append(DCF_NB,dcf_norm_NB)
            DCF_MIN_NB = numpy.append(DCF_MIN_NB,minDCF_NB)
    
    print("size of DCF_MVG, DCF_TC, DCF_BV: ", len(DCF_MVG),len(DCF_TC),len(DCF_NB))
    plt.figure()
    plt.plot(effPriorLogOdds, DCF_MVG, label='DCF_MVG', color='#ADD8E6')
    plt.plot(effPriorLogOdds, DCF_MIN_MVG, label='minDCF_MVG', color='#00008B')
    plt.plot(effPriorLogOdds, DCF_TC, label='DCF_TC', color='#FFB6C1')
    plt.plot(effPriorLogOdds, DCF_MIN_TC, label='minDCF_TC', color='#8B0000')
    plt.plot(effPriorLogOdds, DCF_NB, label='DCF_NB', color='#90EE90')
    plt.plot(effPriorLogOdds, DCF_MIN_NB, label='minDCF_NB', color='#006400')
    plt.ylim([0,1.1])
    plt.xlim([-4,4])
    plt.legend()
    plt.show()

def trainLogReg(DTR,LTR,l,prior=None):
    def logreg_obj(v):

        n = LTR.shape[0]
        w = v[0:-1]
        b =  v[-1]
        ZTR = 2*LTR -1
        S = (vcol(w).T @ DTR + b).ravel()
        epsilon=1 
        if prior:
            nT = numpy.sum(LTR==1)
            nF = n - nT
            epsilon = numpy.where(ZTR==1, prior/nT, (1-prior)/nF)
            
        t1 = (l/2)*numpy.linalg.norm(w) ** 2
        t2 = (1/n) * numpy.sum(epsilon*numpy.logaddexp(0,-ZTR*S)) #output loss shape (d+1,)== [w,b]
        
        J = t1 + t2

        G = -ZTR/(1.0 + numpy.exp(ZTR*S))
        Jw = l*w + (1/n)*numpy.sum(epsilon*(vrow(G)*DTR), axis=1) #(4,)
        Jb = (numpy.sum((epsilon*G)/n)) # scalar
        dJ = numpy.concatenate([Jw, [Jb]])
        
        return J, dJ

    x_min, f_min, d = scipy.optimize.fmin_l_bfgs_b(func = logreg_obj, x0 = numpy.zeros(DTR.shape[0]+1), approx_grad=False,maxfun=50000,factr=100)
    return x_min, f_min, d 


if __name__ == '__main__':

    ## Data loading, visualization, and statistics --Assignment 1

    features, labels = load('trainData.txt')
    labels= labels.flatten()

    features_classT = features[:,labels.flatten()==1]
    features_classF = features[:,labels.flatten()==0]

    #computeStats(features_classT,features_classF)
 
    # PCA, LDA, & Classification --Assignment 2

    ## applying PCA
    PCAdata,_ = PCA_projection(6,features)

    ## applying LDA
    LDAdata, W_LDA = LDA_projection(features,labels)

    ## LDA as a classifier

    ##the split to be used throughout ENTIRE project
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(features, labels)

    '''
    error, accuracy, correct_samples = LDA_Classifier(DTR,LTR,DVAL,LVAL,False,None,True)
    print("LDA: Error, Accuracy, Correct Samples: ",error,accuracy,correct_samples)
    

    # preprocessing with PCA then LDA classification
    error, accuracy, correct_samples = LDA_Classifier(DTR,LTR,DVAL,LVAL,True,6,False)
    print("LDA + PCA : Error, Accuracy, Correct Samples: ",error,accuracy,correct_samples)
    
    # Gaussian Models --Assignment 3
    
    for i in range(features.shape[0]):
        
        featuresT = features[i,labels.flatten()==1].reshape(1,sum(labels.flatten()==1))
        featuresF = features[i,labels.flatten()==0].reshape(1,sum(labels.flatten()==0))
       
        meanT,covT = ML(featuresT)
        meanF,covF = ML(featuresF)

        logNxT = logpdf_GAU_ND(featuresT,meanT,covT)
        logNxF = logpdf_GAU_ND(featuresF,meanF,covF)

        plt.figure()
        plt.hist(featuresT.reshape(sum(labels.flatten()==1)), bins=20, alpha=0.5, label= 'Genuine Fingerprint', color='green', density= True)
        plt.hist(featuresF.reshape(sum(labels.flatten()==0)), bins=20, alpha=0.5, label= 'Fake Fingerprint', color='red', density= True)
        XPlot1 = numpy.linspace(-4, 4, featuresT.shape[1]).reshape(1, featuresT.shape[1])
        XPlot2 = numpy.linspace(-4, 4, featuresF.shape[1]).reshape(1, featuresF.shape[1])
        plt.plot(XPlot1.ravel(), numpy.exp(logpdf_GAU_ND(XPlot1,meanT,covT)), color='green')
        plt.plot(XPlot2.ravel(), numpy.exp(logpdf_GAU_ND(XPlot2,meanF,covF)), color='red')
        plt.legend()
        plt.title('Feature %d' % (i+1))
        plt.show()
    

    ## Assignment 4

    ## MVG

    SPost, likelihoodF_MVG, likelihoodT_MVG = MVG(DTR,LTR,DVAL, 0.5)
    predicts_MVG, llr_scores_MVG = llr(likelihoodF_MVG, likelihoodT_MVG,0)
    accuracyMVG, errorMVG = evaluate_model(SPost, LVAL, predicts_MVG,True)

    ## Tied Covariance
    SPost, likelihoodF_TC, likelihoodT_TC = TiedCov(DTR,LTR,DVAL,0.5)
    predicts_TC, llr_scores_TC = llr(likelihoodF_TC, likelihoodT_TC,0)
    accuracyTC, errorTC = evaluate_model(SPost, LVAL, predicts_TC,True)

    ## Naive Bayes
    SPost, likelihoodF_NB, likelihoodT_NB = NaiveBayes(DTR,LTR,DVAL,0.5)
    predicts_NB, llr_scores_NB = llr(likelihoodF_NB, likelihoodT_NB,0)
    accuracyNB, errorNB = evaluate_model(SPost, LVAL, predicts_NB,True)

    print("MVG: Accuracy, Error: ", accuracyMVG,errorMVG)
    print("TC: Accuracy, Error: ", accuracyTC,errorTC)
    print("NB: Accuracy, Error: ", accuracyNB,errorNB)

    #### ANALYSIS
   
    DF= DTR[:, LTR.flatten()==0]
    DT= DTR[:, LTR.flatten()==1]
    mF, cF = ML(DF)
    mT, cT = ML(DT)
    print("Covariance matrix of True class: ", cT)
    print("Covariance matrix of False class: ", cF)

    CorrT =  cT/ ( vcol(cT.diagonal()**0.5) * vrow(cT.diagonal()**0.5) )
    CorrF =  cF/ ( vcol(cF.diagonal()**0.5) * vrow(cF.diagonal()**0.5) )

    print("Correlation matrix of True class: ", CorrT)
    print("Correlation matrix of False class: ", CorrF)


    ## Discarding the last two features, as they do not satisfy the assumption 
    ## that features can be jointly modeled by Gaussian distributions. 

    ## MVG

    DTR_4_features = DTR[:4,:]
    DVAL_4_features = DVAL[:4,:]
    
    SPost, likelihoodF_MVG, likelihoodT_MVG = MVG(DTR_4_features,LTR,DVAL_4_features, 0.5)
    predicts_MVG,_ = llr(likelihoodF_MVG, likelihoodT_MVG,0)
    accuracyMVG, errorMVG = evaluate_model(SPost, LVAL, predicts_MVG,True)

    ## Tied Covariance
    SPost, likelihoodF_TC, likelihoodT_TC = TiedCov(DTR_4_features,LTR,DVAL_4_features,0.5)
    predicts_TC,_ = llr(likelihoodF_TC, likelihoodT_TC,0)
    accuracyTC, errorTC = evaluate_model(SPost, LVAL, predicts_TC,True)

    ## Naive Bayes
    SPost, likelihoodF_NB, likelihoodT_NB = NaiveBayes(DTR_4_features,LTR,DVAL_4_features,0.5)
    predicts_NB, _ = llr(likelihoodF_NB, likelihoodT_NB,0)
    accuracyNB, errorNB = evaluate_model(SPost, LVAL, predicts_NB,True)


    print("MVG 4 features: Accuracy, Error: ", accuracyMVG,errorMVG)
    print("TC 4 features: Accuracy, Error: ", accuracyTC,errorTC)
    print("NB 4 features: Accuracy, Error: ", accuracyNB,errorNB)

    ## Features 1 and 2
    DTR_12_features = DTR[:3,:]
    DVAL_12_features = DVAL[:3,:]
    
    SPost, likelihoodF_MVG, likelihoodT_MVG = MVG(DTR_12_features,LTR,DVAL_12_features, 0.5)
    predicts_MVG,_ = llr(likelihoodF_MVG, likelihoodT_MVG,0)
    accuracyMVG, errorMVG = evaluate_model(SPost, LVAL, predicts_MVG,True)

    ## Tied Covariance
    SPost, likelihoodF_TC, likelihoodT_TC = TiedCov(DTR_12_features,LTR,DVAL_12_features,0.5)
    predicts_TC,_ = llr(likelihoodF_TC, likelihoodT_TC,0)
    accuracyTC, errorTC = evaluate_model(SPost, LVAL, predicts_TC,True)

    print("MVG features 1 and 2: Accuracy, Error: ", accuracyMVG,errorMVG)
    print("TC features 1 and 2: Accuracy, Error: ", accuracyTC,errorTC)

    ## Features 3 and 4
    DTR_34_features = DTR[2:5,:]
    DVAL_34_features = DVAL[2:5,:]
    
    SPost, likelihoodF_MVG, likelihoodT_MVG = MVG(DTR_34_features,LTR,DVAL_34_features, 0.5)
    predicts_MVG,_ = llr(likelihoodF_MVG, likelihoodT_MVG,0)
    accuracyMVG, errorMVG = evaluate_model(SPost, LVAL, predicts_MVG,True)

    ## Tied Covariance
    SPost, likelihoodF_TC, likelihoodT_TC = TiedCov(DTR_34_features,LTR,DVAL_34_features,0.5)
    predicts_TC,_ = llr(likelihoodF_TC, likelihoodT_TC,0)
    accuracyTC, errorTC = evaluate_model(SPost, LVAL, predicts_TC,True)

    print("MVG features 3 and 4: Accuracy, Error: ", accuracyMVG,errorMVG)
    print("TC features 3 and 4: Accuracy, Error: ", accuracyTC,errorTC)

    ## Applying PCA
    m=5
    DTR_PCA,P = PCA_projection(m, DTR)
    DVAL_PCA = numpy.dot(P.T,DVAL)

    ## MVG

    SPost, likelihoodF_MVG, likelihoodT_MVG = MVG(DTR_PCA,LTR,DVAL_PCA, 0.5)
    predicts_MVG,llr_scores_MVG_PCA = llr(likelihoodF_MVG, likelihoodT_MVG,0)
    accuracyMVG, errorMVG = evaluate_model(SPost, LVAL, predicts_MVG,True)

    ## Tied Covariance
    SPost, likelihoodF_TC, likelihoodT_TC = TiedCov(DTR_PCA,LTR,DVAL_PCA,0.5)
    predicts_TC,llr_scores_TC_PCA = llr(likelihoodF_TC, likelihoodT_TC,0)
    accuracyTC, errorTC = evaluate_model(SPost, LVAL, predicts_TC,True)

    ## Naive Bayes
    SPost, likelihoodF_NB, likelihoodT_NB = NaiveBayes(DTR_PCA,LTR,DVAL_PCA,0.5)
    predicts_NB,llr_scores_NB_PCA = llr(likelihoodF_NB, likelihoodT_NB,0)
    accuracyNB, errorNB = evaluate_model(SPost, LVAL, predicts_NB,True)


    print("MVG+PCA: Accuracy, Error: ", accuracyMVG,errorMVG)
    print("TC+PCA: Accuracy, Error: ", accuracyTC,errorTC)
    print("NB+PCA: Accuracy, Error: ", accuracyNB,errorNB)


    ## Assignment 5 - Lab 7
    pi_tilde1 = effective_prior(0.5,1,9)
    pi_tilde2 = effective_prior(0.5,9,1)
    print(pi_tilde1,pi_tilde2)
    #stronger security (higher false positive cost) corresponds to an equivalent lower prior probability of a legit user

    # now we take our decision by using Bayes Optimal Decision, not by comparing the llr with a threshold 
    predict_MVG05 = Bayes_Decision(llr_scores_MVG, 0.5,1,1)
    predict_MVG09 = Bayes_Decision(llr_scores_MVG, 0.9,1,1)
    predict_MVG01 = Bayes_Decision(llr_scores_MVG, 0.1,1,1)

    predict_TC05 = Bayes_Decision(llr_scores_TC, 0.5,1,1)
    predict_TC09 = Bayes_Decision(llr_scores_TC, 0.9,1,1)
    predict_TC01 = Bayes_Decision(llr_scores_TC, 0.1,1,1)

    predict_NB05 = Bayes_Decision(llr_scores_NB, 0.5,1,1)
    predict_NB09 = Bayes_Decision(llr_scores_NB, 0.9,1,1)
    predict_NB01 = Bayes_Decision(llr_scores_NB, 0.1,1,1)

    #with PCA:
    predict_MVG05_PCA = Bayes_Decision(llr_scores_MVG_PCA, 0.5,1,1)
    predict_MVG09_PCA = Bayes_Decision(llr_scores_MVG_PCA, 0.9,1,1)
    predict_MVG01_PCA = Bayes_Decision(llr_scores_MVG_PCA, 0.1,1,1)

    predict_TC05_PCA= Bayes_Decision(llr_scores_TC_PCA, 0.5,1,1)
    predict_TC09_PCA= Bayes_Decision(llr_scores_TC_PCA, 0.9,1,1)
    predict_TC01_PCA= Bayes_Decision(llr_scores_TC_PCA, 0.1,1,1)

    predict_NB05_PCA= Bayes_Decision(llr_scores_NB_PCA, 0.5,1,1)
    predict_NB09_PCA= Bayes_Decision(llr_scores_NB_PCA, 0.9,1,1)
    predict_NB01_PCA= Bayes_Decision(llr_scores_NB_PCA, 0.1,1,1)

    # DCF: normalized DCF returned by function binary_dcf
    # while minDCF returned by function min_cost
    
    _,DCF_MVG05, _, _ = binary_dcf(vcol(predict_MVG05), LVAL, 0.5,1,1)
    _,DCF_MVG09, _, _ = binary_dcf(vcol(predict_MVG09), LVAL, 0.9,1,1)
    _,DCF_MVG01, _, _ = binary_dcf(vcol(predict_MVG01), LVAL, 0.1,1,1)

    _,DCF_TC05, _, _ = binary_dcf(vcol(predict_TC05), LVAL, 0.5,1,1)
    _,DCF_TC09, _, _ = binary_dcf(vcol(predict_TC09), LVAL, 0.9,1,1)
    _,DCF_TC01, _, _ = binary_dcf(vcol(predict_TC01), LVAL, 0.1,1,1)

    _,DCF_NB05, _, _ = binary_dcf(vcol(predict_NB05), LVAL, 0.5,1,1)
    _,DCF_NB09, _, _ = binary_dcf(vcol(predict_NB09), LVAL, 0.9,1,1)
    _,DCF_NB01, _, _ = binary_dcf(vcol(predict_NB01), LVAL, 0.1,1,1)

    minDCF_MVG05,_,_ = min_cost(vcol(llr_scores_MVG),vcol(numpy.sort(llr_scores_MVG)),LVAL, 0.5,1,1)
    minDCF_MVG09,_,_ = min_cost(vcol(llr_scores_MVG),vcol(numpy.sort(llr_scores_MVG)),LVAL, 0.9,1,1)
    minDCF_MVG01,_,_ = min_cost(vcol(llr_scores_MVG),vcol(numpy.sort(llr_scores_MVG)),LVAL, 0.1,1,1)

    minDCF_TC05,_,_ = min_cost(vcol(llr_scores_TC),vcol(numpy.sort(llr_scores_TC)),LVAL, 0.5,1,1)
    minDCF_TC09,_,_ = min_cost(vcol(llr_scores_TC),vcol(numpy.sort(llr_scores_TC)),LVAL, 0.9,1,1)
    minDCF_TC01,_,_ = min_cost(vcol(llr_scores_TC),vcol(numpy.sort(llr_scores_TC)),LVAL, 0.1,1,1)

    minDCF_NB05,_,_ = min_cost(vcol(llr_scores_NB),vcol(numpy.sort(llr_scores_NB)),LVAL, 0.5,1,1)
    minDCF_NB09,_,_ = min_cost(vcol(llr_scores_NB),vcol(numpy.sort(llr_scores_NB)),LVAL, 0.9,1,1)
    minDCF_NB01,_,_ = min_cost(vcol(llr_scores_NB),vcol(numpy.sort(llr_scores_NB)),LVAL, 0.1,1,1)

    _,DCF_MVG05_PCA, _, _ = binary_dcf(vcol(predict_MVG05_PCA), LVAL, 0.5,1,1)
    _,DCF_MVG09_PCA, _, _ = binary_dcf(vcol(predict_MVG09_PCA), LVAL, 0.9,1,1)
    _,DCF_MVG01_PCA, _, _ = binary_dcf(vcol(predict_MVG01_PCA), LVAL, 0.1,1,1)

    _,DCF_TC05_PCA, _, _ = binary_dcf(vcol(predict_TC05_PCA), LVAL, 0.5,1,1)
    _,DCF_TC09_PCA, _, _ = binary_dcf(vcol(predict_TC09_PCA), LVAL, 0.9,1,1)
    _,DCF_TC01_PCA, _, _ = binary_dcf(vcol(predict_TC01_PCA), LVAL, 0.1,1,1)

    _,DCF_NB05_PCA, _, _ = binary_dcf(vcol(predict_NB05_PCA), LVAL, 0.5,1,1)
    _,DCF_NB09_PCA, _, _ = binary_dcf(vcol(predict_NB09_PCA), LVAL, 0.9,1,1)
    _,DCF_NB01_PCA, _, _ = binary_dcf(vcol(predict_NB01_PCA), LVAL, 0.1,1,1)

    minDCF_MVG05_PCA,_,_ = min_cost(vcol(llr_scores_MVG_PCA),vcol(numpy.sort(llr_scores_MVG_PCA)),LVAL, 0.5,1,1)
    minDCF_MVG09_PCA,_,_ = min_cost(vcol(llr_scores_MVG_PCA),vcol(numpy.sort(llr_scores_MVG_PCA)),LVAL, 0.9,1,1)
    minDCF_MVG01_PCA,_,_ = min_cost(vcol(llr_scores_MVG_PCA),vcol(numpy.sort(llr_scores_MVG_PCA)),LVAL, 0.1,1,1)

    minDCF_TC05_PCA,_,_ = min_cost(vcol(llr_scores_TC_PCA),vcol(numpy.sort(llr_scores_TC_PCA)),LVAL, 0.5,1,1)
    minDCF_TC09_PCA,_,_ = min_cost(vcol(llr_scores_TC_PCA),vcol(numpy.sort(llr_scores_TC_PCA)),LVAL, 0.9,1,1)
    minDCF_TC01_PCA,_,_ = min_cost(vcol(llr_scores_TC_PCA),vcol(numpy.sort(llr_scores_TC_PCA)),LVAL, 0.1,1,1)

    minDCF_NB05_PCA,_,_ = min_cost(vcol(llr_scores_NB_PCA),vcol(numpy.sort(llr_scores_NB_PCA)),LVAL, 0.5,1,1)
    minDCF_NB09_PCA,_,_ = min_cost(vcol(llr_scores_NB_PCA),vcol(numpy.sort(llr_scores_NB_PCA)),LVAL, 0.9,1,1)
    minDCF_NB01_PCA,_,_ = min_cost(vcol(llr_scores_NB_PCA),vcol(numpy.sort(llr_scores_NB_PCA)),LVAL, 0.1,1,1)


    # Printing DCF values
    print("DCF Values:")
    print(f"DCF_MVG05: {DCF_MVG05}")
    print(f"DCF_MVG09: {DCF_MVG09}")
    print(f"DCF_MVG01: {DCF_MVG01}")

    print(f"DCF_TC05: {DCF_TC05}")
    print(f"DCF_TC09: {DCF_TC09}")
    print(f"DCF_TC01: {DCF_TC01}")

    print(f"DCF_NB05: {DCF_NB05}")
    print(f"DCF_NB09: {DCF_NB09}")
    print(f"DCF_NB01: {DCF_NB01}")

    # Printing minDCF values
    print("\nminDCF Values:")
    print(f"minDCF_MVG05: {minDCF_MVG05}")
    print(f"minDCF_MVG09: {minDCF_MVG09}")
    print(f"minDCF_MVG01: {minDCF_MVG01}")

    print(f"minDCF_TC05: {minDCF_TC05}")
    print(f"minDCF_TC09: {minDCF_TC09}")
    print(f"minDCF_TC01: {minDCF_TC01}")

    print(f"minDCF_NB05: {minDCF_NB05}")
    print(f"minDCF_NB09: {minDCF_NB09}")
    print(f"minDCF_NB01: {minDCF_NB01}")


    # Printing DCF values with PCA
    print("\nDCF Values with PCA:")
    print(f"DCF_MVG05_PCA: {DCF_MVG05_PCA}")
    print(f"DCF_MVG09_PCA: {DCF_MVG09_PCA}")
    print(f"DCF_MVG01_PCA: {DCF_MVG01_PCA}")

    print(f"DCF_TC05_PCA: {DCF_TC05_PCA}")
    print(f"DCF_TC09_PCA: {DCF_TC09_PCA}")
    print(f"DCF_TC01_PCA: {DCF_TC01_PCA}")

    print(f"DCF_NB05_PCA: {DCF_NB05_PCA}")
    print(f"DCF_NB09_PCA: {DCF_NB09_PCA}")
    print(f"DCF_NB01_PCA: {DCF_NB01_PCA}")

    # Printing minDCF values with PCA
    print("\nminDCF Values with PCA:")
    print(f"minDCF_MVG05_PCA: {minDCF_MVG05_PCA}")
    print(f"minDCF_MVG09_PCA: {minDCF_MVG09_PCA}")
    print(f"minDCF_MVG01_PCA: {minDCF_MVG01_PCA}")

    print(f"minDCF_TC05_PCA: {minDCF_TC05_PCA}")
    print(f"minDCF_TC09_PCA: {minDCF_TC09_PCA}")
    print(f"minDCF_TC01_PCA: {minDCF_TC01_PCA}")

    print(f"minDCF_NB05_PCA: {minDCF_NB05_PCA}")
    print(f"minDCF_NB09_PCA: {minDCF_NB09_PCA}")
    print(f"minDCF_NB01_PCA: {minDCF_NB01_PCA}")

    # The best model for (0.1,1,1) with PCA is MVG, m=5
    # m=5 and (0.1,1,1) will be our main application

    # Bayes error plots:

    effPriorLogOdds = numpy.linspace(-4, 4, 40)
    bayes_error_plot(effPriorLogOdds, llr_scores_MVG_PCA,llr_scores_TC_PCA,llr_scores_NB_PCA,LVAL)

    '''
    # Assignment 6: Lab 8 - Logistic Regression

    n = LTR.shape[0]
    pi_emp = numpy.sum(LTR==1)/n
    lambdaa = numpy.logspace(-4, 2, 13)

    DCF_LR= []
    minDCF_LR= []

    for l in lambdaa:
            
        x_min, f_min, d = trainLogReg(DTR,LTR,l)
        w = x_min[0:-1]
        b = x_min[-1]
        #scoring the validation samples
        S = (vcol(w).T @ DVAL + b).ravel()
        S_llr = S - numpy.log(pi_emp/(1-pi_emp))

        predicts = Bayes_Decision(S_llr,0.1,1,1)
        _,DCF,_,_ = binary_dcf(vcol(predicts),LVAL, 0.1,1,1)
        minDCF,_,_ = min_cost(vcol(S_llr),vcol(numpy.sort(S_llr)),LVAL, 0.1,1,1)

        DCF_LR.append(DCF)
        minDCF_LR.append(minDCF)

    print("DCF shape: ", len(DCF_LR))
    plt.figure()
    plt.plot(lambdaa, DCF_LR, label='DCF_LR', color='#ADD8E6')
    plt.plot(lambdaa, minDCF_LR, label='minDCF_LR', color='#00008B')
    plt.legend()
    plt.xscale('log', base=10)
    plt.show()

    ## Subset of trainig data
    DTR_sub= DTR[:, ::50]
    LTR_sub= LTR[::50]

    
    n = LTR_sub.shape[0]
    pi_emp = numpy.sum(LTR_sub==1)/n
    lambdaa = numpy.logspace(-4, 2, 13)

    DCF_LR= []
    minDCF_LR= []

    for l in lambdaa:
            
        x_min, f_min, d = trainLogReg(DTR_sub,LTR_sub,l)
        w = x_min[0:-1]
        b = x_min[-1]
        #scoring the validation samples
        S = (vcol(w).T @ DVAL + b).ravel()
        S_llr = S - numpy.log(pi_emp/(1-pi_emp))

        predicts = Bayes_Decision(S_llr,0.1,1,1)
        _,DCF,_,_ = binary_dcf(vcol(predicts),LVAL, 0.1,1,1)
        minDCF,_,_ = min_cost(vcol(S_llr),vcol(numpy.sort(S_llr)),LVAL, 0.1,1,1)

        DCF_LR.append(DCF)
        minDCF_LR.append(minDCF)

    print("DCF shape: ", len(DCF_LR))
    plt.figure()
    plt.plot(lambdaa, DCF_LR, label='DCF_LR', color='#ADD8E6')
    plt.plot(lambdaa, minDCF_LR, label='minDCF_LR', color='#00008B')
    plt.legend()
    plt.xscale('log', base=10)
    plt.show()
