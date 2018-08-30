from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
import numpy as np
import code_pythonsubs as hdp

k = 2
# load data
iris = load_iris()

# initiate PCA and classifier
pca = PCA()
classifier = DecisionTreeClassifier()

# transform / fit
#test_data
ndat,ndim = np.shape(iris.data)
ntest = 20
itest = np.random.choice(np.arange(ndat),ntest,replace=False)
testdat = iris.data[itest,:]
testclas = iris.target[itest]

traindat = np.delete(iris.data,itest,0)
trainclas = np.delete(iris.target,itest)

a = np.zeros((ndat,ndim))
X_transformed = pca.fit_transform(iris.data)
a[:,:] = 1.*X_transformed
XT_train = np.delete(a,itest,0)
classifier.fit(XT_train[:,:k], trainclas)


# transform new data using already fitted pca
# (don't re-fit the pca)
newdata_transformed = pca.transform(testdat)

# predict labels using the trained classifier

pred_labels = classifier.predict(newdata_transformed[:,:k])

#now use my k nearest neighbour
knn_test = hdp.k_test(newdata_transformed[:,:k],XT_train[:,:k],trainclas,distance=2,k=3)


print 'pca aided classification'
fracgood = np.shape(np.where(pred_labels == testclas)[0])[0]/1./ntest
fracgood_k = np.shape(np.where(knn_test == testclas)[0])[0]/1./ntest

for i in range(ntest):
 print pred_labels[i],testclas[i],knn_test[i]
print 'frac good',fracgood,fracgood_k






#now without using pca
classifier = DecisionTreeClassifier()
X_transformed = pca.fit_transform(iris.data)
classifier.fit(traindat[:,:k], trainclas)

pred_labels = classifier.predict(testdat[:,:k])

#now use my k nearest neighbour
knn_test = hdp.k_test(testdat[:,:k],traindat[:,:k],trainclas,distance=2,k=3)

print 'no pca classification'
fracgood = np.shape(np.where(pred_labels == testclas)[0])[0]/1./ntest
fracgood_k = np.shape(np.where(knn_test == testclas)[0])[0]/1./ntest
for i in range(ntest):
 print pred_labels[i],testclas[i],knn_test[i]
print 'frac good',fracgood,fracgood_k




#now test pca against sklearn bench mark
pca = PCA()
X_transformed = pca.fit_transform(iris.data)

evec,eval,fraceval,idsort,newdat,newdatsort = hdp.mypca(iris.data,ndim,diagfolder='',label=[])



