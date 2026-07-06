from sklearn.neighbors import KNeighborsClassifier
from tpami2016_optimal_transport_domain_adaptation.datasets import UspsDataset
from tpami2016_optimal_transport_domain_adaptation.datasets import MnistDataset
import numpy as np
from sklearn.model_selection import train_test_split



# Params grids

dims = [10,20,30,40,50,60,70,80,90]

reg = [10**i for i in range(-3,4)]


method_names = [
    "1NN",
    "PCA",
    "GFK",
    "TSL",
    "JDA",
    "DASVM",
    "OT-exact",
    "OT-entropic",
    "OT-group lasso",
    "OT-laplacian",
    "OT-LpLq"
]
# We repeat the experiments 10 times

scores = np.zeros(len(method_names))

for i in range(10):


    # Digit recognition

    X_usps,y_usps = UspsDataset(seed = i).generate()
    X_mnist,y_mnist = MnistDataset(seed = i).generate()

    # Source : Usps , target : Mnist
    Xs_usps = np.array([[]])
    ys_usps = np.array([])
    for j in range(10):
        indices = np.where(y_usps == i)[0][0:20]
        Xs_usps = np.concatenate([Xs_usps,X_usps[indices,:]],axis = 0)
        ys_usps = np.concatenate([ys_usps,y_usps[indices]])
    indices = np.arange(Xs_usps.shape[0])
    indices = np.random.permutation(indices)
    Xs_usps = Xs_usps[indices]
    ys_usps = ys_usps[indices]
    Xval_mnist, Xtest_mnist, yval_mnist, ytest_mnist = train_test_split(X_mnist,y_mnist , test_size=0.5, random_state=i)

    ## 1NN without DA

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(Xs_usps, ys_usps)
    y_pred = knn.predict(Xtest_mnist)
    scores[0] += accuracy_score(ytest_mnist, y_pred)

    ## 1NN with PCA DA
    
    k_max = 10
    pca_acc = 0 
    for k in dims :
        X_joint = np.vstack([Xs_usps, Xval_mnist])
        
        pca = PCA(n_components=k)  # <-- Ici on utilise k
        pca.fit(X_joint)
        
        Xs_pca = pca.transform(Xs)
        Xt_pca = pca.transform(Xt)

        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(Xs_pca, ys_usps)
        y_pred = knn.predict(Xt_pca)
        if accuracy_score(yval_mnist, y_pred) > pca_acc :
            k_max = k
            pca_acc = accuracy_score(yval_mnist, y_pred) 
    X_joint = np.vstack([Xs_usps, Xtest_mnist])
    pca = PCA(n_components=k_max) 
    pca.fit(X_joint)
    
    Xs_pca = pca.transform(Xs)
    Xt_pca = pca.transform(Xt)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(Xs_pca, ys_usps)
    y_pred = knn.predict(Xt_pca)
    scores[1]+= accuracy_score(ytest_mnist, y_pred) 

    # 1NN with OT

    ns = Xs_usps.shape[0]
    # uniform distributions
    ps = np.ones(ns) / ns
    pt = np.ones(Xtest_mnist.shape[0]) / Xtest_mnist.shape[0]
    C = cost_matrix(Xs_usps, Xtest_mnist)
    Cval = cost_matrix(Xs_usps, Xval_mnist)
    # Exact OT

    exact_plan = ot.emd(ps, pt, C)
    exact_Xs = ns * (exact_plan @ Xtest_mnist)

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(exact_Xs, ys_usps)
    y_pred = knn.predict(Xtest_mnist)
    scores[6] += accuracy_score(ytest_mnist, y_pred)

    # Entropic OT 
    Lambda_max = 10**(-3)
    entr_acc = 0 
    for Lambda in reg : 
        sinkhorn_plan = ot.sinkhorn(ps, pt, Cval, reg=Lambda,numItermax=10000)
        sinkhorn_Xs = ns * (sinkhorn_plan @ Xval_mnist)
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(sinkhorn_Xs, ys_usps)
        y_pred = knn.predict(Xval_mnist)
        if accuracy_score(yval_mnist, y_pred) > entr_acc :
            entr_acc = accuracy_score(yval_mnist, y_pred)
            Lambda_max = Lambda
    sinkhorn_plan = ot.sinkhorn(ps, pt, C, reg=Lambda_max,numItermax=10000)
    sinkhorn_Xs = ns * (sinkhorn_plan @ Xtest_mnist)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(sinkhorn_Xs, ys_usps)
    y_pred = knn.predict(Xtest_mnist)
    scores[7] += accuracy_score(ytest_mnist, y_pred)

    for Lambda in reg:
        for eta in reg :

            gl_plan = ot.da.sinkhorn_l1l2_gl(
                        ps, ys_usps , pt, Cval,
                        Lambda,
                        eta=eta,
                        numItermax=10000
                    )
            gl_Xs = ns * (gl_plan @ Xval_mnist)
            knn = KNeighborsClassifier(n_neighbors=1)
            knn.fit(sinkhorn_Xs, ys_usps)
            y_pred = knn.predict(Xtest_mnist)
            scores[7] += accuracy_score(ytest_mnist, y_pred)




    
    

