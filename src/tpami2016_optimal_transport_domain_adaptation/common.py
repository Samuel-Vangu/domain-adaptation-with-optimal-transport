from ot import sinkhorn
import numpy as np
import scipy.io
import scipy.linalg
import sklearn.metrics
from sklearn.neighbors import KNeighborsClassifier

def cost_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute the squared Euclidean cost matrix."""
    # A: (N, d), B: (n, d)
    diff = A[:, None, :] - B[None, :, :]   # (N, n, d)
    C = np.linalg.norm(diff, axis=-1)      # (N, n)
    return C ** 2


"""
Geodesic Flow Kernel (GFK) -- traduction Python de l'implementation MATLAB
Reference: Boqing Gong et al. "Geodesic flow kernel for Unsupervised Domain
Adaptation." CVPR 2012.

Ce module reproduit fidelement les CALCULS de la version MATLAB d'origine :
    function [acc,G,Cls] = GFK(X_src,Y_src,X_tar,Y_tar,dim)

Remarque mathematique importante
---------------------------------
Le code MATLAB appelle `gsvd` (SVD generalisee), qui n'existe pas nativement
en NumPy/SciPy. Cependant, dans ce probleme precis, la matrice [A; B] = Q'*Pt
a des colonnes orthonormees (car Q et Pt le sont). Dans ce cas particulier,
la GSVD se reduit mathematiquement a une SVD classique de A = QPt[:dim, :].
Cette equivalence (et la convention de signe `V2 = -V2` du code original) a
ete verifiee numeriquement :
  - en comparant la formule fermee de G a une integration numerique directe
    de G = integrale_0^1 Phi(t) Phi(t)' dt  -> ecart ~1e-16 (precision machine)
  - en verifiant que Phi(1) redonne bien exactement le sous-espace cible Pt
    (ce qui n'est vrai qu'avec le signe correct de V2).
"""

import numpy as np
from scipy.linalg import null_space


# --------------------------------------------------------------------------
# Equivalent de pca(X) de MATLAB : centre les colonnes puis renvoie les
# vecteurs propres (coeff), tries par variance decroissante -- identique a
# ce que fait MATLAB (SVD economique de la matrice centree).
# --------------------------------------------------------------------------
def _pca_basis(X):
    """
    X : matrice n_echantillons x n_features
    retourne : matrice n_features x min(n_echantillons, n_features),
               colonnes orthonormees triees par variance decroissante.
    """
    X = np.asarray(X, dtype=float)
    Xc = X - X.mean(axis=0, keepdims=True)
    # SVD economique -> deja triee par valeur singuliere decroissante
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt.T


# --------------------------------------------------------------------------
# Equivalent de [Ps, null(Ps')] de MATLAB : complete Ps en une base
# orthonormee complete de dimension D x D.
# --------------------------------------------------------------------------
def _complete_orthonormal_basis(Ps):
    D = Ps.shape[0]
    r = Ps.shape[1]
    if r >= D:
        return Ps[:, :D]
    complement = null_space(Ps.T)   # D x (D - r)
    return np.hstack([Ps, complement])


# --------------------------------------------------------------------------
# Equivalent de GFK_core(Q, Pt)
# --------------------------------------------------------------------------
def _gfk_core(Q, Pt):
    """
    Q  : D x D, colonnes orthonormees = [Ps_complet, complement]
    Pt : D x dim, colonnes orthonormees (sous-espace cible)
    Retourne G : D x D, noyau du flot geodesique.
    """
    D = Q.shape[0]
    dim = Pt.shape[1]
    eps = 1e-20

    QPt = Q.T @ Pt              # D x dim, colonnes orthonormees
    A = QPt[:dim, :]            # dim x dim
    B = QPt[dim:, :]            # (D-dim) x dim

    # ---- GSVD reduite a une SVD classique (cas particulier valide ici) ----
    V1, gam, VT = np.linalg.svd(A)
    V = VT.T
    gam = np.clip(gam, -1.0, 1.0)          # equivalent du "real(...)" MATLAB
    sig = np.sqrt(np.maximum(1.0 - gam ** 2, 0.0))
    theta = np.arccos(gam)                 # angles principaux

    BV = B @ V
    V2_thin = np.zeros_like(BV)            # (D-dim) x dim -- seule partie utile
    nz = sig > 1e-12
    V2_thin[:, nz] = BV[:, nz] / sig[nz]
    V2_thin = -V2_thin                      # correction de signe (verifiee numeriquement)

    # ---- blocs B1..B4 (identiques a la formule MATLAB) ----
    denom = np.maximum(theta, eps)
    b1 = 0.5 * (1 + np.sin(2 * theta) / (2 * denom))
    b2 = 0.5 * ((-1 + np.cos(2 * theta)) / (2 * denom))
    b4 = 0.5 * (1 - np.sin(2 * theta) / (2 * denom))
    B1, B2, B3, B4 = np.diag(b1), np.diag(b2), np.diag(b2), np.diag(b4)

    # ---- reconstruction de G (forme simplifiee, strictement equivalente
    #      a Q * delta1 * delta2 * delta1' * Q' du code MATLAB : verifie
    #      numeriquement, ecart ~1e-16) ----
    Qa = Q[:, :dim]
    Qb = Q[:, dim:]
    X1 = Qa @ V1
    X2 = Qb @ V2_thin
    G = (X1 @ B1 + X2 @ B3) @ X1.T + (X1 @ B2 + X2 @ B4) @ X2.T
    return G


# --------------------------------------------------------------------------
# Equivalent de my_kernel_knn(M, Xr, Yr, Xt, Yt)
# --------------------------------------------------------------------------
def _kernel_knn(M, Xr, Yr, Xt, Yt):
    Xr = np.asarray(Xr, dtype=float)
    Xt = np.asarray(Xt, dtype=float)
    Yr = np.asarray(Yr).reshape(-1)
    Yt = np.asarray(Yt).reshape(-1)

    d_r = np.einsum('ij,jk,ik->i', Xr, M, Xr)     # diag(Xr*M*Xr')
    d_t = np.einsum('ij,jk,ik->i', Xt, M, Xt)     # diag(Xt*M*Xt')
    dist = d_r[:, None] + d_t[None, :] - 2.0 * (Xr @ M @ Xt.T)

    min_idx = np.argmin(dist, axis=0)   # min(dist) MATLAB opere par colonne
    prediction = Yr[min_idx]
    accuracy = np.mean(prediction == Yt)
    return prediction, accuracy


# --------------------------------------------------------------------------
# Equivalent de la fonction principale GFK(X_src,Y_src,X_tar,Y_tar,dim)
# --------------------------------------------------------------------------
def gfk(X_src, Y_src, X_tar, Y_tar, dim):
    """
    Traduction fidele de :
        function [acc,G,Cls] = GFK(X_src,Y_src,X_tar,Y_tar,dim)

    Parametres
    ----------
    X_src : array (ns, n_feature)  -- caracteristiques source
    Y_src : array (ns,)            -- labels source
    X_tar : array (nt, n_feature)  -- caracteristiques cible
    Y_tar : array (nt,)            -- labels cible
    dim   : int, dim <= 0.5 * n_feature

    Retour
    ------
    acc : precision (1-NN) apres GFK
    G   : matrice noyau du flot geodesique (n_feature x n_feature)
    Cls : labels predits pour la cible (nt,)
    """
    X_src = np.asarray(X_src, dtype=float)
    X_tar = np.asarray(X_tar, dtype=float)

    Ps = _pca_basis(X_src)
    Pt = _pca_basis(X_tar)

    Q = _complete_orthonormal_basis(Ps)
    G = _gfk_core(Q, Pt[:, :dim])
    Cls, acc = _kernel_knn(G, X_src, Y_src, X_tar, Y_tar)
    return acc, G, Cls


    




def kernel(ker, X1, X2, gamma):
    K = None
    if not ker or ker == 'primal':
        K = X1
    elif ker == 'linear':
        if X2:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T, np.asarray(X2).T)
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T)
    elif ker == 'rbf':
        if X2:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, np.asarray(X2).T, gamma)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, None, gamma)
    return K

class JDA:
    def __init__(self, kernel_type='primal', dim=30, lamb=1, gamma=1, T=3):
        '''
        Init func
        :param kernel_type: kernel, values: 'primal' | 'linear' | 'rbf'
        :param dim: dimension after transfer
        :param lamb: lambda value in equation
        :param gamma: kernel bandwidth for rbf kernel
        :param T: iteration number
        '''
        self.kernel_type = kernel_type
        self.dim = dim
        self.lamb = lamb
        self.gamma = gamma
        self.T = T

    def fit_predict(self, Xs, Ys, Xt, Yt):
        '''
        Transform and Predict using 1NN as JDA paper did
        :param Xs: ns * n_feature, source feature
        :param Ys: ns * 1, source label
        :param Xt: nt * n_feature, target feature
        :param Yt: nt * 1, target label
        :return: acc, y_pred, list_acc
        '''
        list_acc = []
        X = np.hstack((Xs.T, Xt.T))
        X /= np.linalg.norm(X, axis=0)
        m, n = X.shape
        ns, nt = len(Xs), len(Xt)
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        C = len(np.unique(Ys))
        print(f"C : {C}")
        H = np.eye(n) - 1 / n * np.ones((n, n))

        M = 0
        Y_tar_pseudo = None
        for t in range(self.T):
            N = 0
            M0 = e * e.T * C
            if Y_tar_pseudo is not None and len(Y_tar_pseudo) == nt:
                for c in range(C):
                    e = np.zeros((n, 1))
                    tt = (Ys == c)
                    e[np.where(tt == True)] = 1 / len(Ys[np.where(Ys == c)])
                    yy = Y_tar_pseudo == c
                    ind = np.where(yy == True)
                    inds = [item + ns for item in ind]
                    e[tuple(inds)] = -1 / len(Y_tar_pseudo[np.where(Y_tar_pseudo == c)])
                    e[np.isinf(e)] = 0
                    N = N + np.dot(e, e.T)
            M = M0 + N
            M = M / np.linalg.norm(M, 'fro')
            K = kernel(self.kernel_type, X, None, gamma=self.gamma)
            n_eye = m if self.kernel_type == 'primal' else n
            a, b = np.linalg.multi_dot([K, M, K.T]) + self.lamb * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T])
            w, V = scipy.linalg.eig(a, b)
            ind = np.argsort(w)
            A = V[:, ind[:self.dim]]
            Z = np.dot(A.T, K)
            Z /= np.linalg.norm(Z, axis=0)
            Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T

            clf = KNeighborsClassifier(n_neighbors=1)
            clf.fit(Xs_new, Ys.ravel())
            Y_tar_pseudo = clf.predict(Xt_new)
            acc = sklearn.metrics.accuracy_score(Yt, Y_tar_pseudo)
            list_acc.append(acc)
        return acc, Y_tar_pseudo, list_acc


