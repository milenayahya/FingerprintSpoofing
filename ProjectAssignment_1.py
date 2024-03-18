import numpy
import matplotlib.pyplot as plt


features = numpy.array([])
labels = numpy.array([])

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
    

if __name__ == '__main__':

    features, labels = load('trainData.txt')

    features_classT = features[:,labels.flatten()==0]
    features_classF = features[:,labels.flatten()==1]

    print(features_classT)
    print(features_classT.shape)
    print(features_classF.shape)

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
 

