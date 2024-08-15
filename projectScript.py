import numpy
import matplotlib.pyplot as plt
import scipy.linalg
import random
import pickle

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

    mu_classF = featuresClassF.mean(1).reshape(featuresClassF.shape[0],1)
    print("mean false class: ", mu_classF)

    var_classT = features_classT.var(1)
    var_classF = featuresClassF.var(1)

    std_classT = features_classT.std(1)
    std_classF = featuresClassF.std(1)

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
    # train LDA using only training data
    LDA_TR, W = LDA_projection(DTR,LTR)
    # project the validation data on the LDA direction
    LDA_VAL = numpy.dot(W.T, DVAL)

    if plot:
        train_true = LDA_TR[:,LTR.flatten()==1]
        train_false = LDA_TR[:,LTR.flatten()==0]
        val_true = LDA_VAL[:,LVAL.flatten()==1]
        val_false = LDA_VAL[:,LVAL.flatten()==0]

        color_genuine = '#4E8098'
        color_fake = '#A31621'
        plt.figure(figsize=(12, 6))  # Adjust figure size as needed

        # Plot for training set
        plt.subplot(1, 2, 1)
        plt.hist(train_true[0], bins=20, alpha=0.5, label='Genuine Fingerprints', color=color_genuine, density=True)
        plt.hist(train_false[0], bins=20, alpha=0.5, label='Fake Fingerprints', color=color_fake, density=True)
        plt.title("LDA Projection on Training Set")
        plt.legend()

        # Plot for validation set
        plt.subplot(1, 2, 2)
        plt.hist(val_true[0], bins=20, alpha=0.5, label='Genuine Fingerprints', color=color_genuine, density=True)
        plt.hist(val_false[0], bins=20, alpha=0.5, label='Fake Fingerprints', color=color_fake, density=True)
        plt.title("LDA Projection on Validation Set")
        plt.legend()

        plt.tight_layout()  # Ensures proper spacing between subplots
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
        dcf, dcf_norm, pfn,pfp = binary_dcf(c,labels,pi,Cfn,Cfp)
       
        res.append(dcf_norm)
        Pfn.append(pfn)
        Pfp.append(pfp)
           
    minDCF = min(res)

    return minDCF,Pfn,Pfp

def dcf_error_array(llr,thresholds,labels,pi,Cfn,Cfp):
    res =[]
    Pfn =[]
    Pfp =[]
    for t in thresholds:
        c= numpy.where(llr<=t,0,1)
        dcf, dcf_norm, pfn,pfp = binary_dcf(c,labels,pi,Cfn,Cfp)
       
        res.append(dcf_norm)
        Pfn.append(pfn)
        Pfp.append(pfp)
           
    return res

def dcf_error_plot(scores1, scores2, scores3, scores4, thresholds):
    plt.figure()
    plt.plot(thresholds,scores1,label='GMM_DCF', color='red')
    plt.plot(thresholds,scores2,label='SVM_DCF', color='blue')
    plt.plot(thresholds,scores3,label='LR_DCF', color='green')
    plt.plot(thresholds,scores4,label='Fusion_DCF', color='purple')
    plt.legend()
    plt.show()
    plt.savefig('eval_dcf_error.png',format='png', dpi=300, bbox_inches='tight')
    plt.close()

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

def trainLogReg(DTR, LTR, l, prior=None):
    def logreg_obj(v):
        n = LTR.shape[0]
        w = v[0:-1]
        b = v[-1]
        ZTR = 2 * LTR - 1
        S = (vcol(w).T @ DTR + b).ravel()
        G = -ZTR / (1.0 + numpy.exp(ZTR * S))
        
        if prior is not None:
            nT = numpy.sum(LTR == 1)
            nF = numpy.sum(LTR == 0)
            epsilon = numpy.where(ZTR == 1, prior / nT, (1 - prior) / nF)
            t2 = numpy.sum(epsilon * numpy.logaddexp(0, -ZTR * S))  # output loss shape (d+1,) == [w,b]
            Jw = l * w + numpy.sum(epsilon * (vrow(G) * DTR), axis=1)   # (d,)
            Jb = numpy.sum(epsilon * G)   # scalar
        else:
            epsilon = numpy.ones(n)
            t2 = numpy.sum(epsilon * numpy.logaddexp(0, -ZTR * S)) /n # output loss shape (d+1,) == [w,b]
            Jw = l * w + numpy.sum(epsilon * (vrow(G) * DTR), axis=1) / n  # (d,)
            Jb = numpy.sum(epsilon * G) / n  # scalar
        
        t1 = (l / 2) * numpy.linalg.norm(w) ** 2
        J = t1 + t2 
        dJ = numpy.concatenate([Jw, [Jb]])
        
        return J, dJ

    x0 = numpy.zeros(DTR.shape[0] + 1)
    x_min, f_min, d = scipy.optimize.fmin_l_bfgs_b(func=logreg_obj, x0=x0, approx_grad=False, maxfun=50000, factr=100)
    return x_min, f_min, d

def quadratic_features(X):
    D,n = X.shape
    Phi = numpy.zeros((D**2+D, n))
    for i in range(n):
        #extract i-th sample
        x_i = X[:,i].reshape(-1,1)
        product = numpy.dot(x_i, x_i.T).flatten()
        Phi[:,i] = numpy.concatenate([product,x_i.flatten()])
    return Phi

def logistic_regression_analysis(DTR, LTR, DVAL, LVAL, prior=None, prior_weight=False, title="DCF_LR", label_prefix="", quadratic=False):
    if quadratic:
        DTR = quadratic_features(DTR)
        DVAL = quadratic_features(DVAL)
        
    n = LTR.shape[0]
    if prior_weight==False:
        prior_scale = numpy.sum(LTR == 1) / n
    else:
        prior_scale= prior
    
    lambdaa = numpy.logspace(-4, 2, 13)

    DCF_LR = []
    minDCF_LR = []

    for l in lambdaa:
        if prior_weight:
            x_min, f_min, d = trainLogReg(DTR, LTR, l, prior)
        else:
            x_min, f_min, d = trainLogReg(DTR, LTR, l)
        w = x_min[0:-1]
        b = x_min[-1]
        # scoring the validation samples
        S = (vcol(w).T @ DVAL + b).ravel()
        S_llr = S - numpy.log(prior_scale / (1 - prior_scale))

        predicts = Bayes_Decision(S_llr, prior, 1, 1)
        _, DCF, _, _ = binary_dcf(vcol(predicts), LVAL, prior, 1, 1)
        minDCF, _, _ = min_cost(vcol(S_llr), vcol(numpy.sort(S_llr)), LVAL, prior, 1, 1)

        DCF_LR.append(DCF)
        minDCF_LR.append(minDCF)

    plt.figure()
    plt.plot(lambdaa, DCF_LR, label=f'{label_prefix}DCF_LR', color='#ADD8E6')
    plt.plot(lambdaa, minDCF_LR, label=f'{label_prefix}minDCF_LR', color='#00008B')
    plt.legend()
    plt.xscale('log', base=10)
    plt.title(title)
    plt.show()

    best_minDCF = numpy.min(minDCF_LR)
    best_minDCF_index = numpy.argmin(minDCF_LR)
    bestDCF = DCF_LR[best_minDCF_index]
    return best_minDCF, bestDCF, minDCF_LR, DCF_LR


def poly_kernel(x1,x2,d,k,c):
    G = numpy.dot(x1.T,x2)
    K = (G+c)**d
    return K+k

def rbf_kernel(x1,x2,gamma,k):
    distance = numpy.linalg.norm(x1-x2)**2
    return (numpy.exp(-gamma*distance)+k)

def rbf_kernel_matrix(D,gamma,k):
    n = D.shape[1] # nb of samples
    K = numpy.zeros((n,n))
    for i in range(n):
        for j in range(n):
            K[i,j]= rbf_kernel(D[:,i],D[:,j],gamma,k)
    return K

def rbf_kernel_scores(DTR, DVAL,alpha_opt,z, gamma,k):
    ntr = DTR.shape[1]
    nval = DVAL.shape[1]
    scores = numpy.zeros(nval)
    for i in range(nval):
        for j in range(ntr):
            scores[i] += alpha_opt[j]*z[j]*rbf_kernel(DTR[:,j],DVAL[:,i],gamma,k)
    return scores

def linear_SVM(DTR,DVAL,LTR,LVAL,k,C,prior):
        
        n = LTR.shape[0]
        nval = LVAL.shape[0]

        bounds = [(0,C) for _ in range(n)]
       # alpha0 = numpy.random.uniform(0,C,n) 

        K= numpy.ones((k,n))
         
        z = numpy.where(LTR==1, 1, -1)

        Dhat = numpy.vstack((DTR,K))
        Ghat = numpy.dot(Dhat.T,Dhat)

        Hhat = Ghat * vcol(z) * vrow(z)
        alpha0 = numpy.zeros(Dhat.shape[1])
        alpha_opt, _, _ = scipy.optimize.fmin_l_bfgs_b(obj_dual_fun,alpha0,fprime=gradient_fun,bounds=bounds,args=(Hhat,),factr=1.0)

        what_opt = (vrow(alpha_opt)*vrow(z)*Dhat).sum(1)
        w, b = what_opt[0:DTR.shape[0]], what_opt[-1] * k
        # #what_opt = numpy.dot(Dhat, alpha_opt * z)
        # what_opt = numpy.zeros(Dhat.shape[0])

        # # Compute the optimized weight vector
        # for i in range(alpha_opt.shape[0]):
        #     what_opt += alpha_opt[i] * z[i] * Dhat[:, i]

        score = (vrow(w)@DVAL + b).ravel()
        predicts = Bayes_Decision(score,prior,1,1)

        accuracy = numpy.sum(LVAL==predicts)/nval
        error = 1 - accuracy
        primal_sol = obj_primal_fun(what_opt,C,z,Dhat)
        dual_sol_minimized = -obj_dual_fun(alpha_opt,Hhat)
        gap = primal_sol-dual_sol_minimized

        # print("primal loss: ",primal_sol)
        # print("dual loss: ",dual_sol_minimized,)
        # print("duality gap: ",gap)
        # print("error: ", error)

        _,DCF,_,_ = binary_dcf(vcol(predicts),LVAL,prior,1,1)
        minDCF,_,_ = min_cost(vcol(score), vcol(numpy.sort(score)),LVAL,prior,1,1)

        # print("DCF: ", DCF)
        # print("Min DCF: ", minDCF)

        return primal_sol, dual_sol_minimized, gap, error, minDCF, DCF

def poly_SVM(DTR,DVAL,LTR,LVAL,d,c,k,C,prior):

    n = LTR.shape[0]
    nval = LVAL.shape[0]
    z = numpy.where(LTR==1, 1, -1)

    bounds = [(0,C) for _ in range(n)]
    alpha0= numpy.zeros(DTR.shape[1]) 
    H_poly = vcol(z)*vrow(z)*poly_kernel(DTR,DTR,d,k,c)

    alpha_opt, _, _ = scipy.optimize.fmin_l_bfgs_b(obj_dual_fun,alpha0,fprime=gradient_fun,bounds=bounds,args=(H_poly,),factr=1.0)

    score = numpy.zeros(LVAL.shape[0])

    for i in range(nval):
        for j in range(n):
            score[i] += alpha_opt[j]*z[j]*poly_kernel(DTR[:,j],DVAL[:,i],d,k,c)

    predicts = Bayes_Decision(score,prior,1,1)

    accuracy = numpy.sum(LVAL==predicts)/nval
    error = 1 - accuracy
    dual_sol_minimized = -obj_dual_fun(alpha_opt,H_poly)

    _,DCF,_,_ = binary_dcf(vcol(predicts),LVAL,prior,1,1)
    minDCF,_,_ = min_cost(vcol(score), vcol(numpy.sort(score)),LVAL,prior,1,1)

    return dual_sol_minimized, error,DCF,minDCF

def rbf_SVM(DTR,DVAL,LTR,LVAL,gamma,k,C,prior):
    
    n = LTR.shape[0]
    nval = LVAL.shape[0]
    z = numpy.where(LTR==1, 1, -1)

    bounds = [(0,C) for _ in range(n)]
    alpha0= numpy.zeros(DTR.shape[1]) 
    H_rbf = vcol(z)*vrow(z)*rbf_kernel_matrix(DTR, gamma,k)

    alpha_opt, _, _ = scipy.optimize.fmin_l_bfgs_b(obj_dual_fun,alpha0,fprime=gradient_fun,bounds=bounds,args=(H_rbf,),factr=1.0)

    # #what_opt = numpy.dot(Dhat, alpha_opt*z)
    # what_opt = numpy.zeros(DTR.shape[0])

    # # Compute the optimized weight vector
    # for i in range(alpha_opt.shape[0]):
    #     what_opt += alpha_opt[i] * z[i] * DTR[:, i]

    scores = rbf_kernel_scores(DTR,DVAL,alpha_opt,z,gamma,k)

    predicts = Bayes_Decision(scores,prior,1,1)

    accuracy = numpy.sum(LVAL==predicts)/nval
    error = 1 - accuracy
    dual_sol_minimized = -obj_dual_fun(alpha_opt,H_rbf)

    _,DCF,_,_ = binary_dcf(vcol(predicts),LVAL,prior,1,1)
    minDCF,_,_ = min_cost(vcol(scores), vcol(numpy.sort(scores)),LVAL,prior,1,1)

    return dual_sol_minimized, error,DCF,minDCF,scores

def obj_dual_fun(alpha,Hhat):
    H = Hhat @ vcol(alpha)
    return (0.5 * (vrow(alpha)@H).ravel() - alpha.sum())

def obj_primal_fun(w,C,z,D):
    return (0.5*numpy.linalg.norm(w)**2 + C*(numpy.maximum(0,1-z*(vrow(w)@D).ravel())).sum())

def gradient_fun(alpha,Hhat):
    n= alpha.size
    return (Hhat@vcol(alpha).ravel() - numpy.ones(n))

def logpdf_GMM(X,gmm):
        S = numpy.zeros((len(gmm),X.shape[1]))
        
        for g, (w, mu, C) in enumerate(gmm):
            S[g, :] = logpdf_GAU_ND(X, mu, C) + numpy.log(w)

        logdens = scipy.special.logsumexp(S,axis=0)
        return S,logdens

def EM(X,gmm, threshold, psi = None, diag=False, tied=False):
    N = X.shape[1]
    LL = [] #log likelihood of all trainig dataset, so sum of LL of all samples, for each step t

    while True:

        # E-step
        S, logdens = logpdf_GMM(X, gmm)
        ll= logdens.mean()# average log likelihood
        LL.append(ll)

        # stop - criterion
        if len(LL) > 1 and numpy.abs(LL[-1] - LL[-2]) < threshold:
            break

        gamma = numpy.exp(S - logdens)

        # M-step
        gmm_new = []

        for i in range(len(gmm)):
            gamma_now = gamma[i]
            Zg = gamma_now.sum()
            Fg = vcol((vrow(gamma_now)*X).sum(1))
            Sg = (vrow(gamma_now)*X) @ X.T     
  
            mu_new = Fg/Zg
            C_new = Sg/Zg - mu_new @ mu_new.T
            w_new = Zg/ N

            if diag:
                C_new = C_new*numpy.eye(X.shape[0])

            gmm_new.append((w_new,mu_new,C_new))

        if tied:
            CT = 0
            for w,mu,C in gmm_new:
                CT += w*C
            
            gmm_new = [(w,mu, CT) for (w,mu,C)in gmm_new]

        ## Constraining eigenvalues
        if psi is not None:
            for i in range(len(gmm_new)):
                w, mu, C = gmm_new[i]
                U, s, _ = numpy.linalg.svd(C)
                s[s < psi] = psi
                C = numpy.dot(U, numpy.diag(s)).dot(U.T)
                gmm_new[i] = (w, mu, C)
                
            
        
        gmm = gmm_new


    return gmm, LL

def LBG_EM(X,threshold, nbComponents,alpha, psi=None, diag=False, tied=False):

    mu, C = ML(X)

    if diag:
        C = C * numpy.eye(X.shape[0])

    if psi is not None:
        U, s, _ = numpy.linalg.svd(C) 
        s[s<psi] = psi
        C = numpy.dot(U, vcol(s)*U.T)

    gmm = [(1.0, mu, C)]

    while (len(gmm)<nbComponents):

        gmm_new = []
        for (w,mu,C) in gmm:
            U, s, Vh = numpy.linalg.svd(C)
            d = U[:, 0:1] * s[0]**0.5 * alpha
            gmm_new.append((0.5*w, mu -d, C))
            gmm_new.append((0.5*w, mu +d, C))

        gmm = gmm_new
        gmm, ll = EM(X,gmm,threshold,psi,diag,tied)

    return gmm

def bayes_error_plot_general(effPriors, scores, labels):
    DCF = []
    minDCF = []

    for p in effPriors:
        pi_tilde = 1/ (1 +numpy.exp(-p))
        # mindcf
        min,_,_ = min_cost(vcol(scores),vcol(numpy.sort(scores)),labels, pi_tilde,1,1)
        # dcf
        pred = Bayes_Decision(scores,pi_tilde,1,1)
        _,actDCF,_,_ = binary_dcf(vcol(pred),labels,pi_tilde,1,1)

        DCF.append(actDCF)
        minDCF.append(min)
    return DCF, minDCF

def dcf_packed(scores,labels, pi, Cfn, Cfp):
    predictions = Bayes_Decision(scores,pi,Cfn,Cfp)
    dcf, dcf_norm,_,_ = binary_dcf(vcol(predictions),labels,pi,Cfn,Cfp)
    return dcf_norm

def minDCF_packed(scores,labels, pi, Cfn, Cfp):
    thresholds = numpy.sort(scores)
    minDCF,Pfn,Pfp = min_cost(vcol(scores),vcol(thresholds),labels,pi,Cfn,Cfp)
    return minDCF

def extract_folds(X, idx, KFold):
    cal = numpy.hstack([X[j::KFold] for j in range(KFold) if j != idx])
    val = X[idx::KFold]
    return cal, val

def Kfold(scores, labels, prior, k):
    calibrated_scores = []
    labels_sys = []

    for fold in range(k):
        
        SCAL, SVAL = extract_folds(scores,fold,k)
        LCAL, LVAL = extract_folds(labels,fold,k)
        x_min, f_min, d = trainLogReg(vrow(SCAL), LCAL,0, prior)
        w = x_min[0:-1]
        b = x_min[-1]


        calibrated_SVAL = (w.T @ vrow(SVAL) + b - numpy.log(prior / (1-prior))).ravel()
        calibrated_scores.append(calibrated_SVAL)
        labels_sys.append(LVAL)

    calibrated_scores = numpy.hstack(calibrated_scores)
    labels_sys = numpy.hstack(labels_sys)

    return calibrated_scores, labels_sys

def Kfold_fusion(scores1, scores2, scores3, labels, prior, k):
    
    combined = list(zip(scores1, scores2, scores3, labels))
    
    # Shuffle the combined list to maintain alignment across models
    random.shuffle(combined)
    
    # Separate the shuffled data back into individual scores and labels
    scores1, scores2, scores3, shuffled_labels = zip(*combined)
    
    # Convert to numpy arrays
    scores1 = numpy.array(scores1)
    scores2 = numpy.array(scores2)
    scores3 = numpy.array(scores3)
    shuffled_labels = numpy.array(shuffled_labels)
    fused_scores =[]
    fused_labels =[]

    for fold in range(k):
        SCAL1, SVAL1 = extract_folds(scores1,fold,k)
        SCAL2, SVAL2 = extract_folds(scores2,fold,k)
        SCAL3, SVAL3 = extract_folds(scores3,fold,k)
        LCAL, LVAL = extract_folds(shuffled_labels, fold, k)

        # Build the training scores "feature" matrix
        SCAL = numpy.vstack([SCAL1, SCAL2, SCAL3])
        
        x_min, f_min, d = trainLogReg(SCAL, LCAL,0, prior)
        w = x_min[0:-1]
        b = x_min[-1]
        SVAL = numpy.vstack([vrow(SVAL1), vrow(SVAL2), vrow(SVAL3)])

        calibrated_SVAL = (w.T @ SVAL + b - numpy.log(prior / (1-prior))).ravel()
        fused_scores.append(calibrated_SVAL)
        fused_labels.append(LVAL)

    fused_scores = numpy.hstack(fused_scores)
    fused_labels = numpy.hstack(fused_labels)

    return fused_scores, fused_labels



if __name__ == '__main__':

    ## Data loading, visualization, and statistics --Assignment 1

    features, labels = load('trainData.txt')
    labels= labels.flatten()
    color_genuine = '#4E8098'
    color_fake = '#A31621'

    # features_classT = features[:,labels.flatten()==1]
    # features_classF = features[:,labels.flatten()==0]

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
    
    # preprocessing with PCA then LDA classification
    error, accuracy, correct_samples = LDA_Classifier(DTR,LTR,DVAL,LVAL,True,1,False)
    
    # Write the results to the file
    with open('results.txt', 'w') as f:
        f.write(f"LDA: Error, Accuracy, Correct Samples: {error}, {accuracy}, {correct_samples}\n")
        f.write(f"LDA + PCA: Error, Accuracy, Correct Samples: {error}, {accuracy}, {correct_samples}\n")

    # Gaussian Models --Assignment 3
        
    fig, axes = plt.subplots(2,3, figsize=(18,12))

    for i in range(features.shape[0]):
        featuresT = features[i, labels.flatten() == 1].reshape(1, sum(labels.flatten() == 1))
        featuresF = features[i, labels.flatten() == 0].reshape(1, sum(labels.flatten() == 0))

        meanT, covT = ML(featuresT)
        meanF, covF = ML(featuresF)

        logNxT = logpdf_GAU_ND(featuresT, meanT, covT)
        logNxF = logpdf_GAU_ND(featuresF, meanF, covF)

        ax = axes[i // 3, i % 3]
        ax.hist(featuresT.flatten(), bins=20, alpha=0.5, label='Genuine Fingerprint', color=color_genuine, density=True)
        ax.hist(featuresF.flatten(), bins=20, alpha=0.5, label='Fake Fingerprint', color=color_fake, density=True)
        
        XPlot1 = numpy.linspace(-4, 4, featuresT.shape[1]).reshape(1, featuresT.shape[1])
        XPlot2 = numpy.linspace(-4, 4, featuresF.shape[1]).reshape(1, featuresF.shape[1])
        
        ax.plot(XPlot1.ravel(), numpy.exp(logpdf_GAU_ND(XPlot1, meanT, covT)), color=color_genuine)
        ax.plot(XPlot2.ravel(), numpy.exp(logpdf_GAU_ND(XPlot2, meanF, covF)), color=color_fake)
        
        ax.legend()
        ax.set_title('Feature %d' % (i + 1))

    fig.tight_layout(pad=5.0)
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

    with open('results.txt', 'a') as f:  
        f.write(f"MVG: Accuracy, Error: {accuracyMVG}, {errorMVG}\n")
        f.write(f"TC: Accuracy, Error: {accuracyTC}, {errorTC}\n")
        f.write(f"NB: Accuracy, Error: {accuracyNB}, {errorNB}\n")
    
    #### ANALYSIS
   
    DF= DTR[:, LTR.flatten()==0]
    DT= DTR[:, LTR.flatten()==1]
    mF, cF = ML(DF)
    mT, cT = ML(DT)

    CorrT =  cT/ ( vcol(cT.diagonal()**0.5) * vrow(cT.diagonal()**0.5) )
    CorrF =  cF/ ( vcol(cF.diagonal()**0.5) * vrow(cF.diagonal()**0.5) )

    with open('results.txt', 'a') as f: 
        f.write("Covariance matrix of True class:\n")
        f.write(f"{numpy.array(cT)}\n\n")
        f.write("Covariance matrix of False class:\n")
        f.write(f"{numpy.array(cF)}\n\n")
        f.write("Correlation matrix of True class:\n")
        f.write(f"{numpy.array(CorrT)}\n\n")
        f.write("Correlation matrix of False class:\n")
        f.write(f"{numpy.array(CorrF)}\n\n")


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


    with open('results.txt', 'a') as f:  
        f.write(f"MVG 4 features: Accuracy, Error: {accuracyMVG}, {errorMVG}\n")
        f.write(f"TC 4 features: Accuracy, Error: {accuracyTC}, {errorTC}\n")
        f.write(f"NB 4 features: Accuracy, Error: {accuracyNB}, {errorNB}\n")

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

    with open('results.txt', 'a') as f:
        f.write(f"MVG features 1 and 2: Accuracy, Error: {accuracyMVG}, {errorMVG}\n")
        f.write(f"TC features 1 and 2: Accuracy, Error: {accuracyTC}, {errorTC}\n")

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

    with open('results.txt', 'a') as f:
        f.write(f"MVG features 3 and 4: Accuracy, Error: {accuracyMVG}, {errorMVG}\n")
        f.write(f"TC features 3 and 4: Accuracy, Error: {accuracyTC}, {errorTC}\n")

  
    ## Applying PCA
    m=1
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


    with open('results.txt', 'a') as f:
        f.write(f"MVG+PCA: Accuracy, Error: {accuracyMVG}, {errorMVG}\n")
        f.write(f"TC+PCA: Accuracy, Error: {accuracyTC}, {errorTC}\n")
        f.write(f"NB+PCA: Accuracy, Error: {accuracyNB}, {errorNB}\n")
        
    
    ## Assignment 5 - Lab 7
    pi_tilde1 = effective_prior(0.5,1,9)
    pi_tilde2 = effective_prior(0.5,9,1)
    print(pi_tilde1,pi_tilde2)
    #stronger security (higher false positive cost) corresponds to an equivalent lower prior probability of a legit user

    
    thresholds = [0.5, 0.9, 0.1]
    models = [
        (llr_scores_MVG, llr_scores_TC, llr_scores_NB),
        (llr_scores_MVG_PCA, llr_scores_TC_PCA, llr_scores_NB_PCA)
    ]

    dcfs = [[], []]
    min_dcfs = [[], []]

    # Loop through models and thresholds
    for idx, (llr_MVG, llr_TC, llr_NB) in enumerate(models):
        for threshold in thresholds:
            # Make predictions
            predict_MVG = Bayes_Decision(llr_MVG, threshold, 1, 1)
            predict_TC = Bayes_Decision(llr_TC, threshold, 1, 1)
            predict_NB = Bayes_Decision(llr_NB, threshold, 1, 1)
            
            # Calculate DCF and minDCF
            _, DCF_MVG, _, _ = binary_dcf(vcol(predict_MVG), LVAL, threshold, 1, 1)
            _, DCF_TC, _, _ = binary_dcf(vcol(predict_TC), LVAL, threshold, 1, 1)
            _, DCF_NB, _, _ = binary_dcf(vcol(predict_NB), LVAL, threshold, 1, 1)
            
            minDCF_MVG, _, _ = min_cost(vcol(llr_MVG), vcol(numpy.sort(llr_MVG)), LVAL, threshold, 1, 1)
            minDCF_TC, _, _ = min_cost(vcol(llr_TC), vcol(numpy.sort(llr_TC)), LVAL, threshold, 1, 1)
            minDCF_NB, _, _ = min_cost(vcol(llr_NB), vcol(numpy.sort(llr_NB)), LVAL, threshold, 1, 1)
            
            # Append results to lists
            dcfs[idx].extend([DCF_MVG, DCF_TC, DCF_NB])
            min_dcfs[idx].extend([minDCF_MVG, minDCF_TC, minDCF_NB])

    labels_dcf = [
        "DCF_MVG05", "DCF_MVG09", "DCF_MVG01",
        "DCF_TC05", "DCF_TC09", "DCF_TC01",
        "DCF_NB05", "DCF_NB09", "DCF_NB01"
    ]

    labels_min_dcf = [
        "minDCF_MVG05", "minDCF_MVG09", "minDCF_MVG01",
        "minDCF_TC05", "minDCF_TC09", "minDCF_TC01",
        "minDCF_NB05", "minDCF_NB09", "minDCF_NB01"
    ]

    # Write results to file
    with open('results.txt', 'a') as f:
        # DCF Values
        f.write("DCF Values:\n")
        for label, value in zip(labels_dcf, dcfs[0]):
            f.write(f"{label}: {value}\n")

        # minDCF Values
        f.write("\nminDCF Values:\n")
        for label, value in zip(labels_min_dcf, min_dcfs[0]):
            f.write(f"{label}: {value}\n")

        # DCF Values with PCA
        f.write("\nDCF Values with PCA:\n")
        for label, value in zip(labels_dcf, dcfs[1]):
            f.write(f"{label}: {value}\n")

        # minDCF Values with PCA
        f.write("\nminDCF Values with PCA:\n")
        for label, value in zip(labels_min_dcf, min_dcfs[1]):
            f.write(f"{label}: {value}\n")
    # The best PCA setup for pi-tilde = 0.1 is no PCA

    # Bayes error plots:

    effPriorLogOdds = numpy.linspace(-4, 4, 40)
    bayes_error_plot(effPriorLogOdds, llr_scores_MVG,llr_scores_TC,llr_scores_NB,LVAL)
    
    
    # Assignment 6: Lab 8 - Logistic Regression

    # Original Data
    best_minDCF, bestDCF,_,_ = logistic_regression_analysis(DTR, LTR, DVAL, LVAL,prior=0.1)
    with open('results.txt', 'a') as f:
        f.write(f"LR - original data: Best minDCF, actualDCF (Original Data): {best_minDCF}, {bestDCF}\n")

    # Subset of training data
    DTR_sub = DTR[:, ::50]
    LTR_sub = LTR[::50]
    logistic_regression_analysis(DTR_sub, LTR_sub, DVAL, LVAL,prior=0.1, title="DCF_LR_sub", label_prefix="sub_")


    # Prior-weighted LR
    pi_prior = 0.1
    best_minDCF_prior, bestDCF_prior,_,_ = logistic_regression_analysis(DTR, LTR, DVAL, LVAL, prior=pi_prior,prior_weight=True, title="DCF_LR_prior", label_prefix="prior_")
    with open('results.txt', 'a') as f:
        f.write(f"LR - prior-weighted: Best minDCF, actualDCF (Original Data): {best_minDCF}, {bestDCF}\n")


    # Quadratic Linear Regression
    _,_,best_minDCF_quad, bestDCF_quad = logistic_regression_analysis(DTR, LTR, DVAL, LVAL, prior=0.1,title="DCF_LR_quad", label_prefix="quad_", quadratic=True)
    with open('results.txt', 'a') as f:
        f.write(f"LR - quadratic: Best minDCF, actualDCF (Original Data): {best_minDCF}, {bestDCF}\n")


    # Pre-processing the data
    mean_TR, cov_TR = ML(DTR)
    var_TR = numpy.var(DTR)
    DTR_z_norm = (DTR - mean_TR) / numpy.sqrt(var_TR)

    eigenvals, eigenvecs = numpy.linalg.eigh(cov_TR)
    A = eigenvecs @ numpy.diag(1.0 / numpy.sqrt(eigenvals)) @ eigenvecs.T

    DTR_z_norm_white = A @ DTR_z_norm
    DVAL_z_norm = (DVAL - mean_TR) / numpy.sqrt(var_TR)
    DVAL_z_norm_white = A @ DVAL_z_norm

    best_minDCF_preproc, bestDCF_preproc,_,_ = logistic_regression_analysis(DTR_z_norm_white, LTR, DVAL_z_norm_white, LVAL,prior=0.1, title="DCF_LR_preproc", label_prefix="preproc_")
    with open('results.txt', 'a') as f:
        f.write(f"LR - pre-processed data: Best minDCF, actualDCF (Original Data): {best_minDCF}, {bestDCF}\n")

    
    ### SVM
    ## Linear SVM

    k=1
    C= numpy.logspace(-5,0,11)
    prior = 0.1

    DCF_linearSVM = []
    minDCF_linearSVM = []
    mu = DTR.mean(1).reshape(DTR.shape[0],1)

    for val in C:
        _,_,_,_,minDCF,DCF = linear_SVM(DTR-mu,DVAL,LTR,LVAL,k,val,prior)
        DCF_linearSVM.append(DCF)
        minDCF_linearSVM.append(minDCF)


    with open('results.txt', 'a') as f:
        f.write(f"minDCF linearSVM: {minDCF_linearSVM}\n")
        f.write(f"dcf linearSVM: {DCF_linearSVM}\n")

    plt.figure()
    plt.plot(C, DCF_linearSVM, label='DCF_linearSVM', color='#ADD8E6')
    plt.plot(C, minDCF_linearSVM, label='minDCF_linearSVM', color='#00008B')
    plt.legend()
    plt.xscale('log', base=10)
    plt.show()


    
    ## Poly-SVM

    prior = 0.1
    d = 2
    c = 1
    k = 0
    C= numpy.logspace(-5,0,11)

    DCF_polySVM = []
    minDCF_polySVM = []

    for val in C:
        _,_,DCF,minDCF = poly_SVM(DTR,DVAL,LTR,LVAL,d,c,k,val,prior)
        DCF_polySVM.append(DCF)
        minDCF_polySVM.append(minDCF)

    with open('results.txt', 'a') as f:
        f.write(f"minDCF polySVM: {minDCF_polySVM}\n")
        f.write(f"dcf polySVM: {DCF_polySVM}\n")

    plt.figure()
    plt.plot(C, DCF_polySVM, label='DCF_polySVM', color='#ADD8E6')
    plt.plot(C, minDCF_polySVM, label='minDCF_polySVM', color='#00008B')
    plt.legend()
    plt.xscale('log', base=10)
    plt.show()
    
    
    
    ## RBF-SVM
    
    prior = 0.1
    k = 1
    gamma = [numpy.exp(-4),numpy.exp(-3),numpy.exp(-2),numpy.exp(-1)]
    C =  numpy.logspace(-3,2,11)

    DCF_rbfSVM = numpy.zeros((len(gamma),len(C)))
    minDCF_rbfSVM = numpy.zeros((len(gamma),len(C)))
    
    i = 0
    for gammaa in gamma:
        DCF_arr = []
        minDCF_arr = []
        for val in C:
            _,_,DCF,minDCF,_ = rbf_SVM(DTR,DVAL,LTR,LVAL,gammaa,k,val,prior)
            DCF_arr.append(DCF)
            minDCF_arr.append(minDCF)
        
        DCF_rbfSVM[i,:] = DCF_arr
        minDCF_rbfSVM[i,:] = minDCF_arr
        i += 1
         
    with open("results.txt", "a") as f:
        plt.figure()

        # Plot and write for gamma = e^{-4}
        plt.plot(C, DCF_rbfSVM[0, :], label=r'actual_DCF_rbfSVM, $\gamma = e^{-4}$', color='#ADD8E6')
        plt.plot(C, minDCF_rbfSVM[0, :], label=r'minDCF_rbfSVM, $\gamma = e^{-4}$', color='#00008B')
        f.write("Gamma = e^{-4}\n")
        for c_value, actual_dcf, min_dcf in zip(C, DCF_rbfSVM[0, :], minDCF_rbfSVM[0, :]):
            f.write(f"C = {c_value}, actual_DCF = {actual_dcf}, min_DCF = {min_dcf}\n")
        f.write("\n")

        # Plot and write for gamma = e^{-3}
        plt.plot(C, DCF_rbfSVM[1, :], label=r'actual_DCF_rbfSVM, $\gamma = e^{-3}$', color='#FFB6C1')
        plt.plot(C, minDCF_rbfSVM[1, :], label=r'minDCF_rbfSVM, $\gamma = e^{-3}$', color='#8B0000')
        f.write("Gamma = e^{-3}\n")
        for c_value, actual_dcf, min_dcf in zip(C, DCF_rbfSVM[1, :], minDCF_rbfSVM[1, :]):
            f.write(f"C = {c_value}, actual_DCF = {actual_dcf}, min_DCF = {min_dcf}\n")
        f.write("\n")

        # Plot and write for gamma = e^{-2}
        plt.plot(C, DCF_rbfSVM[2, :], label=r'actual_DCF_rbfSVM, $\gamma = e^{-2}$', color='#90EE90')
        plt.plot(C, minDCF_rbfSVM[2, :], label=r'minDCF_rbfSVM, $\gamma = e^{-2}$', color='#006400')
        f.write("Gamma = e^{-2}\n")
        for c_value, actual_dcf, min_dcf in zip(C, DCF_rbfSVM[2, :], minDCF_rbfSVM[2, :]):
            f.write(f"C = {c_value}, actual_DCF = {actual_dcf}, min_DCF = {min_dcf}\n")
        f.write("\n")

        # Plot and write for gamma = e^{-1}
        plt.plot(C, DCF_rbfSVM[3, :], label=r'actual_DCF_rbfSVM, $\gamma = e^{-1}$', color='#FFA500')
        plt.plot(C, minDCF_rbfSVM[3, :], label=r'minDCF_rbfSVM, $\gamma = e^{-1}$', color='#FF8C00')
        f.write("Gamma = e^{-1}\n")
        for c_value, actual_dcf, min_dcf in zip(C, DCF_rbfSVM[3, :], minDCF_rbfSVM[3, :]):
            f.write(f"C = {c_value}, actual_DCF = {actual_dcf}, min_DCF = {min_dcf}\n")
        f.write("\n")

    plt.legend()
    plt.show()
    plt.xscale('log', base=10)
    plt.xlabel('C (log scale)')
    plt.ylabel('Actual DCF and minDCF')
    plt.title(r'Actual DCF and minDCF vs C for different $\gamma$ values (RBF SVM)')
    plt.grid(True)
    plt.show()
    
    ## GMM
    alpha = 0.1
    threshold = 10**(-6)
    psi = 0.01

    # full covarinace
    for c in [1,2,4,8,16,32]:
        gmm1 =LBG_EM(DTR[:,LTR==1],threshold,c,alpha,psi)
        gmm0 =LBG_EM(DTR[:,LTR==0],threshold,c,alpha,psi)

        SLLR = logpdf_GMM(DVAL,gmm1)[1] - logpdf_GMM(DVAL,gmm0)[1]
    
        predicts = Bayes_Decision(SLLR, 0.1, 1, 1)
        _, DCF, _, _ = binary_dcf(vcol(predicts), LVAL, 0.1, 1, 1)
        minDCF, _, _ = min_cost(vcol(SLLR), vcol(numpy.sort(SLLR)), LVAL, 0.1, 1, 1)
        with open("results.txt", "a") as file:
            file.write(f'GMM full - minDCF for {c} components: {minDCF:.4f}\n')
            file.write(f'GMM full - actDCF for {c} components: {DCF:.4f}\n')

    # diagonal covariance
    for c in [1,2,4,8,16,32]:
        gmm1 =LBG_EM(DTR[:,LTR==1],threshold,c,alpha,psi,diag=True)
        gmm0 =LBG_EM(DTR[:,LTR==0],threshold,c,alpha,psi,diag=True)

        SLLR = logpdf_GMM(DVAL,gmm1)[1] - logpdf_GMM(DVAL,gmm0)[1]
    
        predicts = Bayes_Decision(SLLR, 0.1, 1, 1)
        _, DCF, _, _ = binary_dcf(vcol(predicts), LVAL, 0.1, 1, 1)
        minDCF, _, _ = min_cost(vcol(SLLR), vcol(numpy.sort(SLLR)), LVAL, 0.1, 1, 1)
        with open("results.txt", "a") as file:
            file.write(f'GMM diagonal - minDCF for {c} components: {minDCF:.4f}\n')
            file.write(f'GMM diagonal - actDCF for {c} components: {DCF:.4f}\n')

    
    
    # Bayes error plots
    # best models:
    
    #best gmm
    gmm1 =LBG_EM(DTR[:,LTR==1],10**(-6),8,0.1,0.01,diag=True)
    gmm0 =LBG_EM(DTR[:,LTR==0],10**(-6),8,0.1,0.01,diag=True)

    with open('gmm1.pkl', 'wb') as f:
        pickle.dump(gmm1, f)

    with open('gmm0.pkl', 'wb') as f:
        pickle.dump(gmm0, f)
    
    
    gmm_best_scores = logpdf_GMM(DVAL,gmm1)[1] - logpdf_GMM(DVAL,gmm0)[1]
    

    #best svm
    _,_,_,_,svm_rbf_best_scores = rbf_SVM(DTR,DVAL,LTR,LVAL,numpy.exp(-2),1,10**1.5,0.1)

    
    # best LR
    DTR_quad, DVAL_quad = quadratic_features(DTR), quadratic_features(DVAL)
    x_min, f_min, d = trainLogReg(DTR_quad, LTR, 10**(-1.5))
    w,b = x_min[0:-1], x_min[-1]
    prior_scale = numpy.sum(LTR == 1) / LTR.shape[0]
    S = (vcol(w).T @ DVAL_quad + b).ravel()
    numpy.save('S_LR.npy', S)

    lr_quad_best_scores = S - numpy.log(prior_scale / (1 - prior_scale))

    numpy.save('gmm_best_scores.npy', gmm_best_scores)
    numpy.save('svm_rbf_best_scores.npy', svm_rbf_best_scores)
    numpy.save('lr_quad_best_scores.npy', lr_quad_best_scores)
    
    
    priors = numpy.linspace(-4, 4, 40)
    dcf_gmm, minDCF_gmm = bayes_error_plot_general(priors, gmm_best_scores,LVAL)
    dcf_svm, minDCF_svm = bayes_error_plot_general(priors, svm_rbf_best_scores,LVAL)
    dcf_LR, minDCF_LR = bayes_error_plot_general(priors, lr_quad_best_scores,LVAL)

    with open('results.txt', 'a') as f:
        f.write(f"DCF GMM: {dcf_gmm}\n")
        f.write(f"minDCF GMM: {minDCF_gmm}\n")
        f.write(f"DCF SVM: {dcf_svm}\n")
        f.write(f"minDCF SVM: {minDCF_svm}\n")
        f.write(f"DCF LR: {dcf_LR}\n")
        f.write(f"minDCF LR: {minDCF_LR}\n")

    plt.figure()
    plt.plot(priors, dcf_gmm, label='dcf_gmm', color='#ADD8E6')
    plt.plot(priors, minDCF_gmm, label='minDCF_gmm', color='#00008B')
    plt.plot(priors, dcf_svm, label='dcf_svm', color='#FFB6C1')
    plt.plot(priors, minDCF_svm, label='minDCF_svm', color='#8B0000')
    plt.plot(priors, dcf_LR, label='DCF_LR', color='#90EE90')
    plt.plot(priors, minDCF_LR, label='minDCF_LR', color='#006400')
    plt.xlim([-4,4])
    plt.legend()
    plt.show()
    
    '''

    # K-FOLD calibration
    # Set random seeds for reproducibility
    random_seed = 42
    random.seed(random_seed)
    numpy.random.seed(random_seed)
    k = 10
    priors = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    '''
    dcf_gmm = []
    dcf_svm = []
    dcf_lr =[]

    gmm_best_scores = numpy.load('gmm_best_scores.npy')
    svm_rbf_best_scores = numpy.load('svm_rbf_best_scores.npy')
    lr_quad_best_scores = numpy.load('lr_quad_best_scores.npy')

    #shuffling the scores
    gmm = list(zip(gmm_best_scores,LVAL))
    svm = list(zip(svm_rbf_best_scores,LVAL))
    lr = list(zip(lr_quad_best_scores,LVAL))
    random.shuffle(gmm)
    random.shuffle(svm)
    random.shuffle(lr)
    gmm_best_scores, gmm_labels = zip(*gmm)
    svm_rbf_best_scores, svm_labels = zip(*svm)
    lr_quad_best_scores, lr_labels = zip(*lr)
    gmm_best_scores = numpy.array(gmm_best_scores)
    gmm_labels = numpy.array(gmm_labels)
    svm_rbf_best_scores = numpy.array(svm_rbf_best_scores)
    svm_labels = numpy.array(svm_labels)
    lr_quad_best_scores = numpy.array(lr_quad_best_scores)
    lr_labels = numpy.array(lr_labels)

    
    for p in priors:
        # train on different priors
        calibrated_scores_gmm, labels_gmm = Kfold(gmm_best_scores,gmm_labels, p,k)
        calibrated_scores_svm, labels_svm = Kfold(svm_rbf_best_scores,svm_labels,p,k)
        calibrated_scores_lr, labels_lr = Kfold(lr_quad_best_scores,lr_labels,p,k)
    
        # evaluate on the target application
        dcf_gmm.append(dcf_packed(vrow(calibrated_scores_gmm),labels_gmm,0.1,1,1))
        dcf_svm.append(dcf_packed(vrow(calibrated_scores_svm),labels_svm,0.1,1,1))
        dcf_lr.append(dcf_packed(vrow(calibrated_scores_lr),labels_lr,0.1,1,1))
    
    # save to plot
    with open('result.txt', 'a') as f:

    # Write DCF values for GMM
        f.write("DCF values for GMM model:\n")
        for i, value in enumerate(dcf_gmm):
            f.write(f"Prior {priors[i]}: {value}\n")
        
        # Write DCF values for SVM
        f.write("DCF values for SVM model:\n")
        for i, value in enumerate(dcf_svm):
            f.write(f"Prior {priors[i]}: {value}\n")
        
        # Write DCF values for LR
        f.write("DCF values for LR model:\n")
        for i, value in enumerate(dcf_lr):
            f.write(f"Prior {priors[i]}: {value}\n")

    # select best performing calibration transformation for every model (lowest  dcf for target application)
    prior_training_gmm = priors[dcf_gmm.index(min(dcf_gmm))]
    prior_training_svm = priors[dcf_svm.index(min(dcf_svm))]
    prior_training_lr = priors[dcf_lr.index(min(dcf_lr))]

    dcf_gmm = min(dcf_gmm)
    dcf_svm = min(dcf_svm)
    dcf_lr = min(dcf_lr)

    minDCF_gmm = minDCF_packed(calibrated_scores_gmm,labels_gmm,0.1,1,1)
    minDCF_svm = minDCF_packed(calibrated_scores_svm,labels_svm,0.1,1,1)
    minDCF_lr = minDCF_packed(calibrated_scores_lr,labels_lr,0.1,1,1)

    # to check calibration
    with open('result.txt', 'a') as f:
        print('writing to file')
        f.write(f"Best prior for training the GMM calibrated model: {prior_training_gmm}\n")
        f.write(f"Best prior for training the SVM calibrated model: {prior_training_svm}\n")
        f.write(f"Best prior for training the LR calibrated model: {prior_training_lr}\n")
        f.write(f"actual DCF of best calibrated GMM model, minDCF of GMM model: {dcf_gmm},{minDCF_gmm}\n")
        f.write(f"actual DCF of best calibrated SVM model, minDCF of SVM model: {dcf_svm},{minDCF_svm}\n")
        f.write(f"actual DCF of best calibrated LR model, minDCF of LR model: {dcf_lr},{minDCF_lr}\n")


    # Bayes Error Plot using best calibrated scores 
    
    # Best Models:
    calibrated_scores_gmm, labels_gmm = Kfold(gmm_best_scores,gmm_labels, prior_training_gmm,k)
    calibrated_scores_svm, labels_svm = Kfold(svm_rbf_best_scores,svm_labels,prior_training_svm,k)
    calibrated_scores_lr, labels_lr = Kfold(lr_quad_best_scores,lr_labels,prior_training_lr,k)

    priors = numpy.linspace(-4,4,40)
    dcf_gmm, minDCF_gmm = bayes_error_plot_general(priors, calibrated_scores_gmm,labels_gmm)
    dcf_svm, minDCF_svm = bayes_error_plot_general(priors, calibrated_scores_svm,labels_svm)
    dcf_LR, minDCF_LR = bayes_error_plot_general(priors, calibrated_scores_lr,labels_lr)

       
    with open('result.txt', 'a') as f:
        f.write(f"DCF calibrated GMM over a range of application priors:\n {dcf_gmm}\n")
        f.write(f"minDCF calibrated GMM over a range of application priors:\n {minDCF_gmm}\n")
        f.write(f"DCF calibrated SVM over a range of application priors:\n {dcf_svm}\n")
        f.write(f"minDCF calibrated SVM over a range of application priors:\n {minDCF_svm}\n")
        f.write(f"DCF calibrated LR over a range of application priors:\n {dcf_LR}\n")
        f.write(f"minDCF calibrated LR over a range of application priors:\n {minDCF_LR}\n")
    
    plt.figure()
    plt.plot(priors, dcf_gmm, label='dcf_gmm', color='#ADD8E6')
    plt.plot(priors, minDCF_gmm, label='minDCF_gmm', color='#00008B')
    plt.plot(priors, dcf_svm, label='dcf_svm', color='#FFB6C1')
    plt.plot(priors, minDCF_svm, label='minDCF_svm', color='#8B0000')
    plt.plot(priors, dcf_LR, label='DCF_LR', color='#90EE90')
    plt.plot(priors, minDCF_LR, label='minDCF_LR', color='#006400')
    plt.xlim([-4,4])
    plt.legend()
    plt.show()

    
    # FUSION
    #unshuffled scores
    gmm_best_scores = numpy.load('gmm_best_scores.npy')
    svm_rbf_best_scores = numpy.load('svm_rbf_best_scores.npy')
    lr_quad_best_scores = numpy.load('lr_quad_best_scores.npy')

    dcf = []
    priors = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for p in priors:
        fused_scores, fused_labels = Kfold_fusion(gmm_best_scores, svm_rbf_best_scores, lr_quad_best_scores, LVAL, p, k)
        dcf.append(dcf_packed(fused_scores, fused_labels, 0.1,1,1))
    
    # best model:
    training_prior = priors[dcf.index(min(dcf))] 
    fused_scores, fused_labels = Kfold_fusion(gmm_best_scores, svm_rbf_best_scores, lr_quad_best_scores, LVAL, training_prior, k)
    best_dcf = min(dcf)
    min_dcf = minDCF_packed(fused_scores,fused_labels,0.1,1,1)

    with open('result.txt', 'a') as f:
        f.write(f"Best prior for training the fused model: {training_prior}\n")
        f.write(f"DCF calibrated fused scores model : {best_dcf}\n")
        f.write(f"minDCF calibrated fused scores model: {min_dcf}\n")
    
    '''
    
    ####### EVALUATION #######
    evalFeatures, evalLabels = load('evalData.txt')
    evalLabels= evalLabels.flatten()

    ## the delivered system is GMM with diag cov and c=8, the scores being calibrated, the calibration transformation 
    ## trained on prior = 0.3. It is trained on the training dataset, but evaluated on our new eval dataset

    with open('gmm1.pkl', 'rb') as f:
        gmm1 = pickle.load(f)

    with open('gmm0.pkl', 'rb') as f:
        gmm0 = pickle.load(f)

    gmm_eval_scores = logpdf_GMM(evalFeatures,gmm1)[1] - logpdf_GMM(evalFeatures,gmm0)[1]

    #shuffling the scores

    gmm = list(zip(gmm_eval_scores,evalLabels))
    random.shuffle(gmm)
    scores,labels = zip(*gmm)
    gmm_scores = numpy.array(scores)
    gmm_labels = numpy.array(labels)

    #calibrating the scores
    calibrated_scores_gmm, labels_gmm = Kfold(gmm_scores,gmm_labels, 0.3,k) # ==>> BEST DELIVERED SYSTEM
    
    numpy.save('gmm_eval_calibrated_scores.npy', calibrated_scores_gmm)
    numpy.save('labels_gmm.npy', labels_gmm)

    #evaluating (point 1)
    dcf = dcf_packed(vrow(calibrated_scores_gmm),labels_gmm,0.1,1,1)
    minDCF = minDCF_packed(calibrated_scores_gmm,labels_gmm,0.1,1,1)
    dcf_bayes_gmm, minDCF_bayes_gmm = bayes_error_plot_general(priors, calibrated_scores_gmm,labels_gmm)

    plt.figure()
    plt.plot(priors, dcf_bayes_gmm, label='dcf_gmm', color='#ADD8E6')
    plt.plot(priors, minDCF_bayes_gmm, label='minDCF_gmm', color='#00008B')
    plt.legend()
    plt.show()
    plt.savefig('delivered_system_bayes.png',format='png')
    plt.close()

    with open('evalResult.txt', 'a') as f:
        f.write(f"actual dcf of eval data using delivered (gmm) system: {dcf}\n")
        f.write(f"minimum dcf of eval data using delivered (gmm) system: {minDCF}\n")


    ## Other best models

    ## Best SVM evaluated on evaludation data
    _,_,_,_,svm_rbf_eval_scores = rbf_SVM(DTR,evalFeatures,LTR,evalLabels,numpy.exp(-2),1,10**1.5,0.1)
    svm = list(zip(svm_rbf_eval_scores,evalLabels))
    random.shuffle(svm)
    scores,labels = zip(*svm)
    svm_scores = numpy.array(scores)
    svm_labels = numpy.array(labels)
    calibrated_scores_svm, labels_svm = Kfold(svm_scores,svm_labels,0.3,k)
    numpy.save('svm_rbf_eval_calibrated_scores.npy', calibrated_scores_svm)
    numpy.save('labels_svm.npy', labels_svm)
    dcf = dcf_packed(vrow(calibrated_scores_svm),labels_svm,0.1,1,1)
    with open('evalResult.txt', 'a') as f:
        f.write(f"actual dcf of eval data using svm system: {dcf}\n")

    ## Best LR evaluated on evaludation data
    S = numpy.load('S_LR.npy')
    prior_scale = numpy.sum(LTR == 1) / LTR.shape[0]
    lr_quad_eval_scores = S - numpy.log(prior_scale / (1 - prior_scale))
    lr = list(zip(lr_quad_eval_scores,evalLabels))
    random.shuffle(lr)
    scores,labels = zip(*lr)
    lr_scores = numpy.array(scores)
    lr_labels = numpy.array(labels)
    calibrated_scores_lr, labels_lr = Kfold(lr_scores,lr_labels,0.9,k)
    numpy.save('lr_quad_eval_scores.npy', calibrated_scores_lr)
    numpy.save('labels_lr.npy', labels_lr)
    dcf = dcf_packed(vrow(calibrated_scores_lr),labels_lr,0.1,1,1)
    with open('evalResult.txt', 'a') as f:
        f.write(f"actual dcf of eval data using lr system: {dcf}\n")


    ## Fusion best model
    fused_scores, fused_labels = Kfold_fusion(gmm_eval_scores, svm_rbf_eval_scores, lr_quad_eval_scores, evalLabels, 0.2, k)
    numpy.save('fused_scores.npy', fused_scores)
    numpy.save('fused_labels.npy', fused_labels)
    dcf = dcf_packed(vrow(fused_scores),fused_labels,0.1,1,1)
    with open('evalResult.txt', 'a') as f:
        f.write(f"actual dcf of eval data using fused system: {dcf}\n")


    # DCF error plots
    thresholds= [calibrated_scores_gmm,calibrated_scores_svm,calibrated_scores_lr,fused_scores]
    array = numpy.concatenate(thresholds)
    array = numpy.sort(array)
    thresholds=array.tolist()



    dcf_gmm = dcf_error_array(calibrated_scores_gmm,thresholds,labels_gmm,0.1,1,1)
    dcf_svm = dcf_error_array(calibrated_scores_svm,thresholds,labels_svm,0.1,1,1)
    dcf_lr = dcf_error_array(calibrated_scores_lr,thresholds,labels_lr,0.1,1,1)
    dcf_fused = dcf_error_array(fused_scores,thresholds,fused_labels,0.1,1,1)

    dcf_error_plot(dcf_gmm,dcf_svm,dcf_lr,dcf_fused,thresholds)


    #actDCF, minDCF, Bayes Error Plot
    #evaluating

    #svm
    dcf = dcf_packed(vrow(calibrated_scores_svm),labels_svm,0.1,1,1)
    minDCF = minDCF_packed(calibrated_scores_svm,labels_svm,0.1,1,1)
    with open('evalResult.txt', 'a') as f:
        f.write(f"actual dcf of eval data using svm system: {dcf}\n")
        f.write(f"minimum dcf of eval data using svm system: {minDCF}\n")

    dcf_bayes_svm, minDCF_bayes_svm = bayes_error_plot_general(priors, calibrated_scores_svm,labels_svm)

    #lr
    dcf = dcf_packed(vrow(calibrated_scores_lr),labels_lr,0.1,1,1)
    minDCF = minDCF_packed(calibrated_scores_lr,labels_lr,0.1,1,1)
    with open('evalResult.txt', 'a') as f:
        f.write(f"actual dcf of eval data using lr system: {dcf}\n")
        f.write(f"minimum dcf of eval data using lr system: {minDCF}\n")

    dcf_bayes_lr, minDCF_bayes_lr = bayes_error_plot_general(priors, calibrated_scores_lr,labels_lr)

    #fusion
    dcf = dcf_packed(vrow(fused_scores),fused_labels,0.1,1,1)
    minDCF = minDCF_packed(fused_scores,fused_labels,0.1,1,1)
    with open('evalResult.txt', 'a') as f:
        f.write(f"actual dcf of eval data using svm system: {dcf}\n")
        f.write(f"minimum dcf of eval data using svm system: {minDCF}\n")

    dcf_bayes_fused, minDCF_bayes_fused= bayes_error_plot_general(priors, fused_scores,fused_labels)

    plt.figure()
    plt.plot(priors, dcf_bayes_gmm, label='dcf_gmm', color='#ADD8E6')
    plt.plot(priors, minDCF_bayes_gmm, label='minDCF_gmm', color='#00008B')
    plt.plot(priors, dcf_bayes_svm, label='dcf_svm', color='#FFB6C1')
    plt.plot(priors, minDCF_bayes_svm, label='minDCF_svm', color='#8B0000')
    plt.plot(priors, dcf_bayes_lr, label='DCF_LR', color='#90EE90')
    plt.plot(priors, minDCF_bayes_lr, label='minDCF_LR', color='#006400')
    plt.plot(priors, dcf_bayes_fused, label='DCF_fused', color='#FAFAD2')
    plt.plot(priors, minDCF_bayes_fused, label='minDCF_fused', color='#DAA520')
    plt.xlim([-4,4])
    plt.legend()
    plt.show()
    plt.savefig('eval_bayes.png',format='png', dpi=300, bbox_inches='tight')
    plt.close()

   