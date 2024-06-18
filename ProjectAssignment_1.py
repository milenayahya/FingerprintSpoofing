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

def logistic_regression_analysis(DTR, LTR, DVAL, LVAL, pi_emp=None, title="DCF_LR", label_prefix="", quadratic=False):
    if quadratic:
        DTR = quadratic_features(DTR)
        DVAL = quadratic_features(DVAL)
        
    n = LTR.shape[0]
    if pi_emp is None:
        pi_emp = numpy.sum(LTR == 1) / n
    
    lambdaa = numpy.logspace(-4, 2, 13)

    DCF_LR = []
    minDCF_LR = []

    for l in lambdaa:
        x_min, f_min, d = trainLogReg(DTR, LTR, l)
        w = x_min[0:-1]
        b = x_min[-1]
        # scoring the validation samples
        S = (vcol(w).T @ DVAL + b).ravel()
        S_llr = S - numpy.log(pi_emp / (1 - pi_emp))

        predicts = Bayes_Decision(S_llr, 0.1, 1, 1)
        _, DCF, _, _ = binary_dcf(vcol(predicts), LVAL, 0.1, 1, 1)
        minDCF, _, _ = min_cost(vcol(S_llr), vcol(numpy.sort(S_llr)), LVAL, 0.1, 1, 1)

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

def linear_SVM(DTR,DVAL,LTR,LVAL,k,C):
        
        n = LTR.shape[0]
        nval = LVAL.shape[0]

        bounds = [(0,C) for _ in range(n)]
        alpha0 = numpy.random.uniform(0,C,n) 

        K= numpy.ones((k,n))
        Kval = numpy.ones((k,nval))
         
        z = numpy.where(LTR==1, 1, -1)

        Dhat = numpy.vstack((DTR,K))
        Ghat = numpy.dot(Dhat.T,Dhat)
    
        Hhat = Ghat * numpy.outer(z,z)

        alpha_opt, _, _ = scipy.optimize.fmin_l_bfgs_b(obj_dual_fun,alpha0,fprime=gradient_fun,bounds=bounds,args=(Hhat,),factr=1.0)

        #what_opt = numpy.dot(Dhat, alpha_opt * z)
        what_opt = numpy.zeros(Dhat.shape[0])

        # Compute the optimized weight vector
        for i in range(alpha_opt.shape[0]):
            what_opt += alpha_opt[i] * z[i] * Dhat[:, i]
       
        Dval_hat = numpy.vstack((DVAL,Kval))

        score = numpy.dot(what_opt.T, Dval_hat)
        predicts = numpy.where(score>0,1,0)

        accuracy = numpy.sum(LVAL==predicts)/nval
        error = 1 - accuracy
        primal_sol = obj_primal_fun(what_opt,C,z,Dhat)
        dual_sol_minimized = -obj_dual_fun(alpha_opt,Hhat)
        gap = primal_sol-dual_sol_minimized

        _,DCF,_,_ = binary_dcf(vcol(predicts),LVAL,0.5,1,1)
        minDCF,_,_ = min_cost(vcol(score), vcol(numpy.sort(score)),LVAL,0.5,1,1)

        return primal_sol, dual_sol_minimized, gap, error, minDCF, DCF


def poly_SVM(DTR,DVAL,LTR,LVAL,d,c,k,C):

    n = LTR.shape[0]
    nval = LVAL.shape[0]
    z = numpy.where(LTR==1, 1, -1)

    bounds = [(0,C) for _ in range(n)]
    alpha0 = numpy.random.uniform(0,C,n) 
    H_poly = numpy.outer(z,z)*poly_kernel(DTR,DTR,d,k,c)

    alpha_opt, _, _ = scipy.optimize.fmin_l_bfgs_b(obj_dual_fun,alpha0,fprime=gradient_fun,bounds=bounds,args=(H_poly,),factr=1.0)

    #what_opt = numpy.dot(Dhat, alpha_opt*z)
    what_opt = numpy.zeros(DTR.shape[0])

    # Compute the optimized weight vector
    for i in range(alpha_opt.shape[0]):
        what_opt += alpha_opt[i] * z[i] * DTR[:, i]

    score = numpy.zeros(LVAL.shape[0])

    for i in range(nval):
        for j in range(n):
            score[i] += alpha_opt[j]*z[j]*poly_kernel(DTR[:,j],DVAL[:,i],d,k,c)

    predicts = numpy.where(score>0,1,0)

    accuracy = numpy.sum(LVAL==predicts)/nval
    error = 1 - accuracy
    dual_sol_minimized = -obj_dual_fun(alpha_opt,H_poly)

    _,DCF,_,_ = binary_dcf(vcol(predicts),LVAL,0.5,1,1)
    minDCF,_,_ = min_cost(vcol(score), vcol(numpy.sort(score)),LVAL,0.5,1,1)

    return dual_sol_minimized, error,DCF,minDCF

def rbf_SVM(DTR,DVAL,LTR,LVAL,gamma,k,C):
    
    n = LTR.shape[0]
    nval = LVAL.shape[0]
    z = numpy.where(LTR==1, 1, -1)

    bounds = [(0,C) for _ in range(n)]
    alpha0 = numpy.random.uniform(0,C,n) 
    H_rbf = numpy.outer(z,z)*rbf_kernel_matrix(DTR, gamma,k)

    alpha_opt, _, _ = scipy.optimize.fmin_l_bfgs_b(obj_dual_fun,alpha0,fprime=gradient_fun,bounds=bounds,args=(H_rbf,),factr=1.0)

    #what_opt = numpy.dot(Dhat, alpha_opt*z)
    what_opt = numpy.zeros(DTR.shape[0])

    # Compute the optimized weight vector
    for i in range(alpha_opt.shape[0]):
        what_opt += alpha_opt[i] * z[i] * DTR[:, i]

    scores = rbf_kernel_scores(DTR,DVAL,alpha_opt,z,gamma,k)

    predicts = numpy.where(scores>0,1,0)

    accuracy = numpy.sum(LVAL==predicts)/nval
    error = 1 - accuracy
    dual_sol_minimized = -obj_dual_fun(alpha_opt,H_rbf)

    _,DCF,_,_ = binary_dcf(vcol(predicts),LVAL,0.5,1,1)
    minDCF,_,_ = min_cost(vcol(scores), vcol(numpy.sort(scores)),LVAL,0.5,1,1)

    return dual_sol_minimized, error,DCF,minDCF

def obj_dual_fun(alpha,Hhat):
    return (0.5 * numpy.dot(alpha, numpy.dot(Hhat, alpha)) - numpy.sum(alpha))

def obj_primal_fun(w,C,z,D):
    n = z.size
    ones = numpy.ones((1,n))
    return (0.5*numpy.linalg.norm(w)**2 + C*(numpy.maximum(0,ones-z*(numpy.dot(w.T,D)))).sum())

def gradient_fun(alpha,Hhat):
    n= alpha.size
    return (numpy.dot(Hhat,alpha) - numpy.ones((1,n))).reshape(n,)


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
    print("LDA: Error, Accuracy, Correct Samples: ",error,accuracy,correct_samples)
    
    # preprocessing with PCA then LDA classification
    error, accuracy, correct_samples = LDA_Classifier(DTR,LTR,DVAL,LVAL,True,1,False)
    print("LDA + PCA : Error, Accuracy, Correct Samples: ",error,accuracy,correct_samples)

    # Gaussian Models --Assignment 3
        
    fig, axes = plt.subplots(2,3, figsize=(18,12))
    fig.tight_layout(pad=5.0)

    for i in range(features.shape[0]):
        featuresT = features[i, labels.flatten() == 1].reshape(1, sum(labels.flatten() == 1))
        featuresF = features[i, labels.flatten() == 0].reshape(1, sum(labels.flatten() == 0))

        meanT, covT = ML(featuresT)
        meanF, covF = ML(featuresF)

        logNxT = logpdf_GAU_ND(featuresT, meanT, covT)
        logNxF = logpdf_GAU_ND(featuresF, meanF, covF)

        ax = axes[i]
        ax.hist(featuresT.flatten(), bins=20, alpha=0.5, label='Genuine Fingerprint', color=color_genuine, density=True)
        ax.hist(featuresF.flatten(), bins=20, alpha=0.5, label='Fake Fingerprint', color=color_fake, density=True)
        
        XPlot1 = numpy.linspace(-4, 4, featuresT.shape[1]).reshape(1, featuresT.shape[1])
        XPlot2 = numpy.linspace(-4, 4, featuresF.shape[1]).reshape(1, featuresF.shape[1])
        
        ax.plot(XPlot1.ravel(), numpy.exp(logpdf_GAU_ND(XPlot1, meanT, covT)), color=color_genuine)
        ax.plot(XPlot2.ravel(), numpy.exp(logpdf_GAU_ND(XPlot2, meanF, covF)), color=color_fake)
        
        ax.legend()
        ax.set_title('Feature %d' % (i + 1))

    plt.show()
    '''
    
    '''
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
    # print("Covariance matrix of True class: ", cT)
    # print("Covariance matrix of False class: ", cF)

    CorrT =  cT/ ( vcol(cT.diagonal()**0.5) * vrow(cT.diagonal()**0.5) )
    CorrF =  cF/ ( vcol(cF.diagonal()**0.5) * vrow(cF.diagonal()**0.5) )

    # print("Correlation matrix of True class: ", CorrT)
    # print("Correlation matrix of False class: ", CorrF)


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

    # The best PCA setup for pi-tilde = 0.1 is no PCA

    # Bayes error plots:

    effPriorLogOdds = numpy.linspace(-4, 4, 40)
    bayes_error_plot(effPriorLogOdds, llr_scores_MVG,llr_scores_TC,llr_scores_NB,LVAL)
    
    
    # Assignment 6: Lab 8 - Logistic Regression

    # Original Data
    best_minDCF, bestDCF,_,_ = logistic_regression_analysis(DTR, LTR, DVAL, LVAL)
    print("Best minDCF, actualDCF (Original Data):", best_minDCF, bestDCF)

    # Subset of training data
    DTR_sub = DTR[:, ::50]
    LTR_sub = LTR[::50]
    logistic_regression_analysis(DTR_sub, LTR_sub, DVAL, LVAL, title="DCF_LR_sub", label_prefix="sub_")


    # Prior-weighted LR
    pi_prior = numpy.sum(LVAL == 1) / LVAL.shape[0]
    best_minDCF_prior, bestDCF_prior,_,_ = logistic_regression_analysis(DTR, LTR, DVAL, LVAL, pi_emp=pi_prior, title="DCF_LR_prior", label_prefix="prior_")
    print("Best minDCF, actualDCF (Prior-weighted):", best_minDCF_prior, bestDCF_prior)

    # Quadratic Linear Regression
    _,_,best_minDCF_quad, bestDCF_quad = logistic_regression_analysis(DTR, LTR, DVAL, LVAL, title="DCF_LR_quad", label_prefix="quad_", quadratic=True)
    print(" minDCF, actualDCF (Quadratic):", best_minDCF_quad,bestDCF_quad)

    # Pre-processing the data
    mean_TR, cov_TR = ML(DTR)
    var_TR = numpy.var(DTR)
    DTR_z_norm = (DTR - mean_TR) / numpy.sqrt(var_TR)

    eigenvals, eigenvecs = numpy.linalg.eigh(cov_TR)
    A = eigenvecs @ numpy.diag(1.0 / numpy.sqrt(eigenvals)) @ eigenvecs.T

    DTR_z_norm_white = A @ DTR_z_norm
    DVAL_z_norm = (DVAL - mean_TR) / numpy.sqrt(var_TR)
    DVAL_z_norm_white = A @ DVAL_z_norm

    best_minDCF_preproc, bestDCF_preproc,_,_ = logistic_regression_analysis(DTR_z_norm_white, LTR, DVAL_z_norm_white, LVAL, title="DCF_LR_preproc", label_prefix="preproc_")
    print("Best minDCF,actualDCF (Pre-processed):", best_minDCF_preproc, bestDCF_preproc)

    '''

    ### SVM
    ## Linear SVM

    k=1
    C= numpy.logspace(-5,0,11)

    DCF_linearSVM = []
    minDCF_linearSVM = []
    mu = DTR.mean(1).reshape(DTR.shape[0],1)

    for val in C:
        _,_,_,_,minDCF,DCF = linear_SVM(DTR,DVAL,LTR,LVAL,k,val)
        DCF_linearSVM.append(DCF)
        minDCF_linearSVM.append(minDCF)
    print("minDCF", minDCF_linearSVM)
    print("dcf", DCF_linearSVM)

    plt.figure()
    plt.plot(C, DCF_linearSVM, label='DCF_linearSVM', color='#ADD8E6')
    plt.plot(C, minDCF_linearSVM, label='minDCF_linearSVM', color='#00008B')
    plt.legend()
    plt.xscale('log', base=10)
    plt.show()

    

    ## Poly-SVM

    d = 2
    c = 1
    k = 0
    C= numpy.logspace(-5,0,11)

    DCF_polySVM = []
    minDCF_polySVM = []

    for val in C:
        _,_,DCF,minDCF = poly_SVM(DTR,DVAL,LTR,LVAL,d,c,k,val)
        DCF_polySVM.append(DCF)
        minDCF_polySVM.append(minDCF)
    
    print("minDCF", minDCF_polySVM)
    print("dcf", DCF_polySVM)
    plt.figure()
    plt.plot(C, DCF_polySVM, label='DCF_polySVM', color='#ADD8E6')
    plt.plot(C, minDCF_polySVM, label='minDCF_polySVM', color='#00008B')
    plt.legend()
    plt.xscale('log', base=10)
    plt.show()
    '''
    ## RBF-SVM
    '''
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
            _,_,DCF,minDCF = rbf_SVM(DTR,DVAL,LTR,LVAL,gammaa,k,val)
            DCF_arr.append(DCF)
            minDCF_arr.append(minDCF)
        
        DCF_rbfSVM[i,:] = DCF_arr
        minDCF_rbfSVM[i,:] = minDCF_arr
        i += 1
         
    plt.figure()
    plt.plot(C, DCF_rbfSVM[0, :], label=r'actual_DCF_rbfSVM, $\gamma = e^{-4}$', color='#ADD8E6')
    plt.plot(C, minDCF_rbfSVM[0, :], label=r'minDCF_rbfSVM, $\gamma = e^{-4}$', color='#00008B')
    plt.plot(C, DCF_rbfSVM[1, :], label=r'actual_DCF_rbfSVM, $\gamma = e^{-3}$', color='#FFB6C1')
    plt.plot(C, minDCF_rbfSVM[1, :], label=r'minDCF_rbfSVM, $\gamma = e^{-3}$', color='#8B0000')
    plt.plot(C, DCF_rbfSVM[2, :], label=r'actual_DCF_rbfSVM, $\gamma = e^{-2}$', color='#90EE90')
    plt.plot(C, minDCF_rbfSVM[2, :], label=r'minDCF_rbfSVM, $\gamma = e^{-2}$', color='#006400')
    plt.plot(C, DCF_rbfSVM[3, :], label=r'actual_DCF_rbfSVM, $\gamma = e^{-1}$', color='#FFA500')
    plt.plot(C, minDCF_rbfSVM[3, :], label=r'minDCF_rbfSVM, $\gamma = e^{-1}$', color='#FF8C00')
    plt.legend()
    plt.xscale('log', base=10)
    plt.xlabel('C (log scale)')
    plt.ylabel('Actual DCF and minDCF')
    plt.title(r'Actual DCF and minDCF vs C for different $\gamma$ values (RBF SVM)')
    plt.grid(True)
    plt.show()
    