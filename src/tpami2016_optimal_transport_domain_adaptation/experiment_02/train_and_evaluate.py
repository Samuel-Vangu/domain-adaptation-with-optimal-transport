"""
Modular domain adaptation experiment with progress logging.
Runs multiple methods on a source→target pair and displays a results table.
Easy to extend with new datasets and methods.
"""


import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import ot
import matplotlib.pyplot as plt
import pandas as pd
import skada
from functools import partial
from scipy.linalg import eigh


# ----- Custom imports (adjust paths to your own libraries) -----
from tpami2016_optimal_transport_domain_adaptation.datasets import UspsDataset, MnistDataset, PieDataset, CaltechOfficeDataset
from tpami2016_optimal_transport_domain_adaptation.common import gfk, JDA
# If you use Transfer Subspace Learning, import it (e.g. from tllib or a custom module)
from skada import TransferSubspaceLearning


# ============================================================
# 1. Helper – 1‑NN accuracy
# ============================================================
def knn_accuracy(X_train, y_train, X_test, y_test, n_neighbors=1):
    """Train a k‑NN classifier and return accuracy on test set."""
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    return accuracy_score(y_test, y_pred)


# ============================================================
# 2. Definition of each adaptation method
#    Each function takes (Xs, ys, Xt_val, yt_val, Xt_test, yt_test)
#    and returns the test accuracy.
# ============================================================

def method_baseline_1nn(Xs, ys, Xt_val, yt_val, Xt_test, yt_test):
    """No adaptation: train on source, test on target."""
    return knn_accuracy(Xs, ys, Xt_test, yt_test)


def method_pca(Xs, ys, Xt_val, yt_val, Xt_test, yt_test, dims):
    """PCA subspace alignment, tune n_components on validation."""
    best_dim = dims[0]
    best_val_acc = 0.0
    print("    [PCA] Tuning n_components...")
    for k in dims:
        joint = np.vstack([Xs, Xt_val])
        pca = PCA(n_components=k)
        pca.fit(joint)
        Xs_pca = pca.transform(Xs)
        Xt_val_pca = pca.transform(Xt_val)
        val_acc = knn_accuracy(Xs_pca, ys, Xt_val_pca, yt_val)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_dim = k
        print(f"      dim={k:3d}  val_acc={val_acc:.4f}  best so far: {best_val_acc:.4f} (dim={best_dim})")
    # Final evaluation
    joint_test = np.vstack([Xs, Xt_test])
    pca = PCA(n_components=best_dim)
    pca.fit(joint_test)
    Xs_pca = pca.transform(Xs)
    Xt_test_pca = pca.transform(Xt_test)
    return knn_accuracy(Xs_pca, ys, Xt_test_pca, yt_test)


def method_gfk(Xs, ys, Xt_val, yt_val, Xt_test, yt_test, dims):
    """Geodesic Flow Kernel, tune dimension on validation."""
    best_dim = dims[0]
    best_val_acc = 0.0
    print("    [GFK] Tuning dimension...")
    for k in dims:
        val_acc, _, _ = gfk(Xs, ys, Xt_val, yt_val, dim=k)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_dim = k
        print(f"      dim={k:3d}  val_acc={val_acc:.4f}  best so far: {best_val_acc:.4f} (dim={best_dim})")
    final_acc, _, _ = gfk(Xs, ys, Xt_test, yt_test, dim=best_dim)
    return final_acc


from skada import TransferSubspaceLearning
from sklearn.metrics import accuracy_score
import numpy as np

def method_tsl(
    Xs,
    ys,
    Xt_val,
    yt_val,
    Xt_test,
    yt_test,
    dims,
    regs,
    verbose=True,
):
    """
    Evaluate Transfer Subspace Learning (TSL) using the full pipeline
    (TransferSubspaceLearningAdapter + KNeighborsClassifier, k=1).

    Hyperparameters are selected on the validation target domain.
    The final model is then learned using all available unlabeled
    target samples (validation + test) and evaluated on the test set.

    Parameters
    ----------
    Xs : ndarray (n_source, n_features)
    ys : ndarray (n_source,)
    Xt_val : ndarray (n_val, n_features)
    yt_val : ndarray (n_val,)
    Xt_test : ndarray (n_test, n_features)
    yt_test : ndarray (n_test,)
    dims : list[int]          # candidate subspace dimensions
    regs : list[float]        # candidate regularization parameters µ
    verbose : bool

    Returns
    -------
    test_accuracy : float
    best_k : int
    best_mu : float
    """
    ys = ys.astype(np.int64)

    best_acc = -1.0
    best_k = None
    best_mu = None

    if verbose:
        print("    [TSL] Hyperparameter search (pipeline)")

    # ----------------------------------------------------------------------
    # 1. Validation : select best (k, mu) using Xt_val
    # ----------------------------------------------------------------------
    for k in dims:
        for mu in regs:
            # Concatenate source and validation target
            X = np.vstack((Xs, Xt_val))
            y = np.concatenate((
                ys,
                np.full(len(Xt_val), -1, dtype=np.int64)   # target labels unknown
            ))
            sample_domain = np.concatenate((
                np.ones(len(Xs), dtype=int),               # source
                -np.ones(len(Xt_val), dtype=int)           # target
            ))

            # Build the full pipeline (adapter + 1‑NN)
            clf = TransferSubspaceLearning(
                n_components=k,
                mu=mu,
                base_method="pca"
            )

            # Fit on source+target (projection learned, 1‑NN trained on source)
            clf.fit(X, y, sample_domain=sample_domain)

            # Predict on validation target
            y_pred = clf.predict(
                Xt_val,
                sample_domain=-np.ones(len(Xt_val), dtype=int)
            )
            val_acc = accuracy_score(yt_val, y_pred)

            if val_acc > best_acc:
                best_acc = val_acc
                best_k = k
                best_mu = mu

            if verbose:
                print(f"      k={k:3d} mu={mu:.1e} val={val_acc:.4f}")

    if verbose:
        print(f"    Best parameters: k={best_k}, mu={best_mu:.1e}")

    # ----------------------------------------------------------------------
    # 2. Final model using all available target data (validation + test)
    # ----------------------------------------------------------------------
    Xt_all = np.vstack((Xt_val, Xt_test))
    X = np.vstack((Xs, Xt_all))
    y = np.concatenate((
        ys,
        np.full(len(Xt_all), -1, dtype=np.int64)
    ))
    sample_domain = np.concatenate((
        np.ones(len(Xs), dtype=int),
        -np.ones(len(Xt_all), dtype=int)
    ))

    clf_final = TransferSubspaceLearning(
        n_components=best_k,
        mu=best_mu,
        base_method="pca"
    )
    clf_final.fit(X, y, sample_domain=sample_domain)

    # Evaluate on the test set
    y_pred_test = clf_final.predict(
        Xt_test,
        sample_domain=-np.ones(len(Xt_test), dtype=int)
    )
    test_acc = accuracy_score(yt_test, y_pred_test)

    if verbose:
        print(f"    Test accuracy = {test_acc:.4f}")

    return test_acc


def method_jda(Xs, ys, Xt_val, yt_val, Xt_test, yt_test, dims, regs):
    """Joint Distribution Adaptation, tune dim and lambda on validation."""
    # Shift labels to 1..C if your JDA implementation requires it (common bug)
    ys_shifted = ys 
    yt_val_shifted = yt_val 
    yt_test_shifted = yt_test

    best_k = dims[0]
    best_lamb = regs[0]
    best_val_acc = 0.0
    print("    [JDA] Tuning dim and lambda...")
    for k in dims:
        for lam in regs:
            val_acc, _, _ = JDA(kernel_type='linear', dim=k, lamb=lam).fit_predict(
                    Xs, ys_shifted, Xt_val, yt_val_shifted)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_k = k
                best_lamb = lam
            print(f"      dim={k:3d}, lambda={lam:.0e}  val_acc={val_acc:.4f}  best so far: {best_val_acc:.4f} (dim={best_k}, lambda={best_lamb:.0e})")
    # Final evaluation on test
    final_acc, _, _ = JDA(kernel_type='linear', dim=best_k, lamb=best_lamb).fit_predict(
            Xs, ys_shifted, Xt_test, yt_test_shifted)

def method_ot_exact(Xs, ys, Xt_val, yt_val, Xt_test, yt_test, **kwargs):
    """Exact OT with no hyper‑parameter (use uniform distributions)."""
    ns = Xs.shape[0]
    nt_test = Xt_test.shape[0]
    ps = np.ones(ns) / ns
    pt_test = np.ones(nt_test) / nt_test
    C_test = ot.dist(Xs, Xt_test, metric='sqeuclidean')
    plan = ot.emd(ps, pt_test, C_test)
    transported_Xs = ns * (plan @ Xt_test)
    return knn_accuracy(transported_Xs, ys, Xt_test, yt_test)


def method_ot_entropic(Xs, ys, Xt_val, yt_val, Xt_test, yt_test, regs):
    """Entropic OT, tune entropy reg on validation."""
    ns = Xs.shape[0]
    nt_val = Xt_val.shape[0]
    nt_test = Xt_test.shape[0]
    ps = np.ones(ns) / ns
    pt_val = np.ones(nt_val) / nt_val
    pt_test = np.ones(nt_test) / nt_test
    C_val = ot.dist(Xs, Xt_val, metric='sqeuclidean')
    C_val /= np.median(C_val)   # ou C_val /= C_val.max()
    C_test = ot.dist(Xs, Xt_test, metric='sqeuclidean')
    C_test /= np.median(C_test)

    best_reg = regs[0]
    best_val_acc = 0.0
    print("    [OT-entropic] Tuning reg...")
    for reg in regs:
        plan = ot.sinkhorn(ps, pt_val, C_val, reg=reg)
        transported_Xs = ns * (plan @ Xt_val)
        val_acc = knn_accuracy(transported_Xs, ys, Xt_val, yt_val)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_reg = reg
        print(f"      reg={reg:.0e}  val_acc={val_acc:.4f}  best so far: {best_val_acc:.4f} (reg={best_reg:.0e})")
    plan_test = ot.sinkhorn(ps, pt_test, C_test, reg=best_reg)
    transported_Xs_test = ns * (plan_test @ Xt_test)
    return knn_accuracy(transported_Xs_test, ys, Xt_test, yt_test)


def method_ot_group_lasso(Xs, ys, Xt_val, yt_val, Xt_test, yt_test, regs):
    """OT with group lasso regularization, tune lambda and eta on validation."""
    ns = Xs.shape[0]
    ps = np.ones(ns) / ns
    pt_val = np.ones(Xt_val.shape[0]) / Xt_val.shape[0]
    pt_test = np.ones(Xt_test.shape[0]) / Xt_test.shape[0]
    C_val = ot.dist(Xs, Xt_val, metric='sqeuclidean')
    C_val /= np.median(C_val)   # ou C_val /= C_val.max()
    C_test = ot.dist(Xs, Xt_test, metric='sqeuclidean')
    C_test /= np.median(C_test)

    best_lam = regs[0]
    best_eta = regs[0]
    best_val_acc = 0.0
    print("    [OT-group lasso] Tuning lambda and eta...")
    for lam in regs:
        for eta in regs:
            plan = ot.da.sinkhorn_l1l2_gl(ps, ys, pt_val, C_val, lam, eta=eta)
            transported_Xs = ns * (plan @ Xt_val)
            val_acc = knn_accuracy(transported_Xs, ys, Xt_val, yt_val)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_lam = lam
                best_eta = eta
            print(f"      lambda={lam:.0e}, eta={eta:.0e}  val_acc={val_acc:.4f}  best so far: {best_val_acc:.4f} (lambda={best_lam:.0e}, eta={best_eta:.0e})")
    plan_test = ot.da.sinkhorn_l1l2_gl(ps, ys, pt_test, C_test, best_lam, eta=best_eta)
    transported_Xs_test = ns * (plan_test @ Xt_test)
    return knn_accuracy(transported_Xs_test, ys, Xt_test, yt_test)


def method_ot_laplacian(Xs, ys, Xt_val, yt_val, Xt_test, yt_test, regs):
    """OT with Laplacian regularization, tune eta on validation."""
    ns = Xs.shape[0]
    ps = np.ones(ns) / ns
    pt_val = np.ones(Xt_val.shape[0]) / Xt_val.shape[0]
    pt_test = np.ones(Xt_test.shape[0]) / Xt_test.shape[0]
    C_val = ot.dist(Xs, Xt_val, metric='sqeuclidean')
    C_test = ot.dist(Xs, Xt_test, metric='sqeuclidean')

    best_eta = regs[0]
    best_val_acc = 0.0
    print("    [OT-laplacian] Tuning eta...")
    for eta in regs:
        plan = ot.da.emd_laplace(ps, pt_val, Xs, Xt_val, C_val, eta=eta)
        transported_Xs = ns * (plan @ Xt_val)
        val_acc = knn_accuracy(transported_Xs, ys, Xt_val, yt_val)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_eta = eta
        print(f"      eta={eta:.0e}  val_acc={val_acc:.4f}  best so far: {best_val_acc:.4f} (eta={best_eta:.0e})")
    plan_test = ot.da.emd_laplace(ps, pt_test, Xs, Xt_test, C_test, eta=best_eta)
    transported_Xs_test = ns * (plan_test @ Xt_test)
    return knn_accuracy(transported_Xs_test, ys, Xt_test, yt_test)


def method_ot_lpl1(Xs, ys, Xt_val, yt_val, Xt_test, yt_test, regs):
    """OT with LpL1 regularization, tune lambda and eta on validation."""
    ns = Xs.shape[0]
    ps = np.ones(ns) / ns
    pt_val = np.ones(Xt_val.shape[0]) / Xt_val.shape[0]
    pt_test = np.ones(Xt_test.shape[0]) / Xt_test.shape[0]
    C_val = ot.dist(Xs, Xt_val, metric='sqeuclidean')
    C_test = ot.dist(Xs, Xt_test, metric='sqeuclidean')

    best_lam = regs[0]
    best_eta = regs[0]
    best_val_acc = 0.0
    print("    [OT-LpL1] Tuning lambda and eta...")
    for lam in regs:
        for eta in regs:
            plan = ot.da.sinkhorn_lpl1_mm(ps, ys, pt_val, C_val, reg=lam, eta=eta)
            transported_Xs = ns * (plan @ Xt_val)
            val_acc = knn_accuracy(transported_Xs, ys, Xt_val, yt_val)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_lam = lam
                best_eta = eta
            print(f"      lambda={lam:.0e}, eta={eta:.0e}  val_acc={val_acc:.4f}  best so far: {best_val_acc:.4f} (lambda={best_lam:.0e}, eta={best_eta:.0e})")
    plan_test = ot.da.sinkhorn_lpl1_mm(ps, ys, pt_test, C_test, reg=best_lam, eta=best_eta)
    transported_Xs_test = ns * (plan_test @ Xt_test)
    return knn_accuracy(transported_Xs_test, ys, Xt_test, yt_test)

def semi_supervised_regularization_entropic(G, ys, yt_labeled, idx_labeled, penalty=1e10, eps=1e-10):
    """
    Regularization: penalty for (source, labeled target) pairs with same class + negative entropy.
    idx_labeled: indices of labeled target samples in the full target set.
    """
    # Penalty only on columns corresponding to labeled targets
    M = np.zeros_like(G)
    for j, idx in enumerate(idx_labeled):
        col = G[:, idx]
        mask = (ys == yt_labeled[j])   # source samples with same class
        M[mask, idx] = penalty
    term_penalty = np.sum(M * G)

    # Entropic term
    G_safe = np.clip(G, eps, None)
    entropic_term = np.sum(G_safe * np.log(G_safe))
    return term_penalty - entropic_term

def semi_supervised_gradient_entropic(G, ys, yt_labeled, idx_labeled, penalty=1e10, eps=1e-10):
    """
    Gradient of the combined regularization.
    """
    M = np.zeros_like(G)
    for j, idx in enumerate(idx_labeled):
        col = G[:, idx]
        mask = (ys == yt_labeled[j])
        M[mask, idx] = penalty
    grad_penalty = M
    grad_entropy = - (np.log(G + eps) + 1.0)
    return grad_penalty + grad_entropy


def method_ot_entropic_semi_supervised(Xs, ys, Xt_labeled, yt_labeled,
                                       Xt_val, yt_val, Xt_test, yt_test, regs):
    """Semi-supervised entropic OT with penalty only on labeled target."""
    # Combine all target samples
    Xt_all_val = np.vstack([Xt_labeled, Xt_val])
    Xt_all_test = np.vstack([Xt_labeled, Xt_test])   # labeled also available at test time? Usually yes.

    ns = Xs.shape[0]
    ps = np.ones(ns) / ns
    pt_val = np.ones(Xt_all_val.shape[0]) / Xt_all_val.shape[0]
    pt_test = np.ones(Xt_all_test.shape[0]) / Xt_all_test.shape[0]

    C_val = ot.dist(Xs, Xt_all_val, metric='sqeuclidean')
    C_test = ot.dist(Xs, Xt_all_test, metric='sqeuclidean')

    # Indices of labeled samples in the combined target sets
    idx_labeled_val = np.arange(len(Xt_labeled))
    idx_labeled_test = np.arange(len(Xt_labeled))

    # Prepare partials
    f_reg_val = partial(semi_supervised_regularization_entropic,
                        ys=ys, yt_labeled=yt_labeled, idx_labeled=idx_labeled_val)
    df_reg_val = partial(semi_supervised_gradient_entropic,
                         ys=ys, yt_labeled=yt_labeled, idx_labeled=idx_labeled_val)

    best_reg = regs[0]
    best_val_acc = 0.0
    print("    [OT-entropic semi-sup] Tuning reg...")
    for reg in regs:
        plan_val = ot.optim.cg(ps, pt_val, C_val, reg, f_reg_val, df_reg_val, numItermax=100)
        # Transport only source to the unlabeled validation part (skip labeled)
        transported_Xs = ns * (plan_val[:, idx_labeled_val[-1]+1:] @ Xt_val)
        val_acc = knn_accuracy(transported_Xs, ys, Xt_val, yt_val)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_reg = reg
        print(f"      reg={reg:.0e}  val_acc={val_acc:.4f}")

    # Final model on test
    f_reg_test = partial(semi_supervised_regularization_entropic,
                         ys=ys, yt_labeled=yt_labeled, idx_labeled=idx_labeled_test)
    df_reg_test = partial(semi_supervised_gradient_entropic,
                          ys=ys, yt_labeled=yt_labeled, idx_labeled=idx_labeled_test)
    plan_test = ot.optim.cg(ps, pt_test, C_test, best_reg, f_reg_test, df_reg_test, numItermax=100)
    transported_Xs_test = ns * (plan_test[:, idx_labeled_test[-1]+1:] @ Xt_test)
    return knn_accuracy(transported_Xs_test, ys, Xt_test, yt_test)


def group_lasso_regularization_semi(G, ys, yt_labeled, idx_labeled):
    """Group Lasso only on labeled target columns."""
    reg = 0.0
    unique_classes = np.unique(ys)
    for cl in unique_classes:
        I_cl = np.where(ys == cl)[0]
        for j, idx in enumerate(idx_labeled):
            # Only penalize if source class == target class
            if yt_labeled[j] == cl:
                reg += np.linalg.norm(G[I_cl, idx])
    return reg

def group_lasso_gradient_semi(G, ys, yt_labeled, idx_labeled):
    """Gradient of the semi-supervised Group Lasso."""
    grad = np.zeros_like(G)
    unique_classes = np.unique(ys)
    for cl in unique_classes:
        I_cl = np.where(ys == cl)[0]
        for j, idx in enumerate(idx_labeled):
            if yt_labeled[j] == cl:
                group = G[I_cl, idx]
                norm = np.linalg.norm(group)
                if norm > 0:
                    grad[I_cl, idx] = group / norm
    return grad

def method_ot_group_lasso_semi_supervised(Xs, ys, Xt_labeled, yt_labeled,
                                          Xt_val, yt_val, Xt_test, yt_test, regs):
    ns = Xs.shape[0]
    Xt_all_val = np.vstack([Xt_labeled, Xt_val])
    Xt_all_test = np.vstack([Xt_labeled, Xt_test])
    ps = np.ones(ns) / ns
    pt_val = np.ones(Xt_all_val.shape[0]) / Xt_all_val.shape[0]
    pt_test = np.ones(Xt_all_test.shape[0]) / Xt_all_test.shape[0]

    C_val = ot.dist(Xs, Xt_all_val, metric='sqeuclidean')
    C_test = ot.dist(Xs, Xt_all_test, metric='sqeuclidean')

    idx_labeled_val = np.arange(len(Xt_labeled))
    idx_labeled_test = np.arange(len(Xt_labeled))

    f_reg = partial(group_lasso_regularization_semi,
                    ys=ys, yt_labeled=yt_labeled, idx_labeled=idx_labeled_val)
    df_reg = partial(group_lasso_gradient_semi,
                     ys=ys, yt_labeled=yt_labeled, idx_labeled=idx_labeled_val)

    best_eta = regs[0]
    best_val_acc = 0.0
    print("    [OT-group lasso semi-sup] Tuning eta...")
    for eta in regs:
        plan_val = ot.optim.cg(ps, pt_val, C_val, eta, f_reg, df_reg, numItermax=100)
        transported_Xs = ns * (plan_val[:, idx_labeled_val[-1]+1:] @ Xt_val)
        val_acc = knn_accuracy(transported_Xs, ys, Xt_val, yt_val)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_eta = eta
        print(f"  eta={eta:.0e}  val_acc={val_acc:.4f}")

    f_reg_test = partial(group_lasso_regularization_semi,
                         ys=ys, yt_labeled=yt_labeled, idx_labeled=idx_labeled_test)
    df_reg_test = partial(group_lasso_gradient_semi,
                          ys=ys, yt_labeled=yt_labeled, idx_labeled=idx_labeled_test)
    plan_test = ot.optim.cg(ps, pt_test, C_test, best_eta, f_reg_test, df_reg_test, numItermax=100)
    transported_Xs_test = ns * (plan_test[:, idx_labeled_test[-1]+1:] @ Xt_test)
    return knn_accuracy(transported_Xs_test, ys, Xt_test, yt_test)




def mmdt_fit_transform(Xs, ys, Xt, k, lambda_reg):
    """
    Exact reimplementation of MMDT (linear) from the ICLR 2013 paper.
    Xs : source features (ns, d)
    ys : source labels (ns,)
    Xt : target features (nt, d)
    k : subspace dimension
    lambda_reg : trade-off parameter (mu in the paper)
    """
    d = Xs.shape[1]
    # Center data
    Xall = np.vstack([Xs, Xt])
    mean_all = Xall.mean(axis=0)
    Xs_c = Xs - mean_all
    Xt_c = Xt - mean_all
    
    ns, nt = Xs.shape[0], Xt.shape[0]
    # MMD matrix (marginal)
    M = np.zeros((ns+nt, ns+nt))
    M[:ns, :ns] = 1/(ns*ns) * np.ones((ns, ns))
    M[:ns, ns:] = -1/(ns*nt) * np.ones((ns, nt))
    M[ns:, :ns] = -1/(ns*nt) * np.ones((nt, ns))
    M[ns:, ns:] = 1/(nt*nt) * np.ones((nt, nt))
    
    # Scatter matrix within source classes
    classes = np.unique(ys)
    Sw = np.zeros((d, d))
    for c in classes:
        Xc = Xs_c[ys == c]
        if len(Xc) > 1:
            Sw += (Xc.T @ Xc)
    
    # Regularised within-scatter
    A = Sw + lambda_reg * np.eye(d)
    
    # MMD term
    Xall_c = np.vstack([Xs_c, Xt_c])
    B = Xall_c.T @ M @ Xall_c
    
    # Solve generalized eigenvalue problem: B v = lambda A v
    eigvals, eigvecs = eigh(B, A)
    # Select k eigenvectors with smallest eigenvalues
    idx = np.argsort(eigvals)[:k]
    W = eigvecs[:, idx]
    
    Xs_proj = Xs_c @ W
    Xt_proj = Xt_c @ W
    return Xs_proj, Xt_proj

def method_mmdt(Xs, ys, Xt_val, yt_val, Xt_test, yt_test, dims, regs, verbose=True):
    ys = ys.astype(np.int64)
    best_acc = -1.0
    best_k = None
    best_mu = None

    if verbose:
        print("    [MMDT] Hyperparameter search")
    for k in dims:
        for mu in regs:
            Xs_proj, Xt_val_proj = mmdt_fit_transform(Xs, ys, Xt_val, k, mu)
            knn = KNeighborsClassifier(n_neighbors=1)
            knn.fit(Xs_proj, ys)
            acc = accuracy_score(yt_val, knn.predict(Xt_val_proj))
            if acc > best_acc:
                best_acc = acc
                best_k = k
                best_mu = mu
            if verbose:
                print(f"      k={k:3d} mu={mu:.1e} val={acc:.4f}")
    if verbose:
        print(f"    Best: k={best_k}, mu={best_mu:.1e}")

    # Final model on all target data
    Xt_all = np.vstack([Xt_val, Xt_test])
    Xs_proj, Xt_all_proj = mmdt_fit_transform(Xs, ys, Xt_all, best_k, best_mu)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(Xs_proj, ys)
    Xt_test_proj = Xt_all_proj[-len(Xt_test):]
    test_acc = accuracy_score(yt_test, knn.predict(Xt_test_proj))
    if verbose:
        print(f"    Test accuracy = {test_acc:.4f}")
    return test_acc, best_k, best_mu




# ============================================================
# 3. Experiment runner for one source→target pair
# ============================================================
def run_experiment(source_data, target_data, methods, n_repetitions=10):
    """
    Run all methods on a source→target pair.
    source_data: callable (seed) -> (Xs, ys)
    target_data: callable (seed) -> (Xt, yt)
    methods: dict {name: callable(Xs,ys,Xt_val,yt_val,Xt_test,yt_test)}
    Returns dict {method_name: average_accuracy}
    """
    method_names = list(methods.keys())
    scores = {name: 0.0 for name in method_names}

    for seed in range(n_repetitions):
        print(f"\n{'='*60}")
        print(f" REPETITION {seed+1}/{n_repetitions} (seed={seed})")
        print(f"{'='*60}")

        # Load data
        Xs, ys = source_data(seed)
        Xt, yt = target_data(seed)

        # Select 20 examples per class from source (as in original)
        Xs_sub, ys_sub = [], []
        rng = np.random.default_rng(seed)

        for class_label in np.unique(ys):
            idx_all = np.where(ys == class_label)[0]
            idx = rng.choice(idx_all, size=20, replace=False)
            Xs_sub.append(Xs[idx])
            ys_sub.append(ys[idx])
        Xs_sub = np.vstack(Xs_sub)
        ys_sub = np.concatenate(ys_sub)
        perm = np.random.permutation(len(Xs_sub))
        Xs_sub = Xs_sub[perm]
        ys_sub = ys_sub[perm]

        # Split target into validation and test (50/50)
        Xt_val, Xt_test, yt_val, yt_test = train_test_split(
                Xt, yt, test_size=0.5, random_state=seed)

        # Evaluate each method
        for name, method in methods.items():
                print(f"  METHOD: {name}")
                acc = method(Xs_sub, ys_sub, Xt_val, yt_val, Xt_test, yt_test)
                scores[name] += acc
                print(f"  => Test accuracy this repetition: {acc:.4f}\n")

    # Average
    for name in scores:
        scores[name] /= n_repetitions
    return scores

import numpy as np
from sklearn.model_selection import train_test_split

def run_experiment_semi_supervised(source_data, target_data, methods, n_repetitions=10):
    """
    Run all methods on a source→target pair in a semi-supervised setting.

    For each repetition:
      1. Source subsampled to 20 examples per class.
      2. 3 labeled target examples per class are extracted.
         These are *removed* from the target pool (they will not be in val/test).
      3. The rest of the target is split into validation and test (50/50).
      4. For semi-supervised OT methods ("Semi-supervised" in name):
            - receive (Xs, ys, Xt_labeled, yt_labeled, Xt_val, yt_val, Xt_test, yt_test)
            - they must include Xt_labeled in the transport matrix (penalty applies there)
            - but evaluation is done on Xt_val / Xt_test only.
      5. For all other methods:
            - the labeled target examples are added to the source training set.
            - they receive (Xs_aug, ys_aug, Xt_val, yt_val, Xt_test, yt_test)
    """
    method_names = list(methods.keys())
    scores = {name: 0.0 for name in method_names}

    for seed in range(n_repetitions):
        print(f"\n{'='*60}")
        print(f" REPETITION {seed+1}/{n_repetitions} (seed={seed})")
        print(f"{'='*60}")

        Xs, ys = source_data(seed)
        Xt, yt = target_data(seed)

        # ---- 1. Subsample source: 20 examples per class ----
        Xs_sub, ys_sub = [], []
        rng = np.random.default_rng(seed)
        for cl in np.unique(ys):
            idx_all = np.where(ys == cl)[0]
            idx = rng.choice(idx_all, size=20, replace=False)
            Xs_sub.append(Xs[idx])
            ys_sub.append(ys[idx])
        Xs_src = np.vstack(Xs_sub)
        ys_src = np.concatenate(ys_sub)

        # ---- 2. Extract 3 labeled target examples per class (same for all methods) ----
        Xt_labeled, yt_labeled = [], []
        Xt_rest, yt_rest = [], []
        for cl in np.unique(yt):
            idx_all = np.where(yt == cl)[0]
            idx_chosen = rng.choice(idx_all, size=3, replace=False)
            Xt_labeled.append(Xt[idx_chosen])
            yt_labeled.append(yt[idx_chosen])
            idx_rest = np.setdiff1d(idx_all, idx_chosen)
            Xt_rest.append(Xt[idx_rest])
            yt_rest.append(yt[idx_rest])
        Xt_labeled = np.vstack(Xt_labeled)
        yt_labeled = np.concatenate(yt_labeled)
        Xt_rest = np.vstack(Xt_rest)
        yt_rest = np.concatenate(yt_rest)

        # Split the unlabeled target into validation / test (50/50)
        Xt_val, Xt_test, yt_val, yt_test = train_test_split(
            Xt_rest, yt_rest, test_size=0.5, random_state=seed
        )

        # ---- 3. Evaluate each method ----
        for name, method in methods.items():
            print(f"  METHOD: {name}")

            if "Semi-supervised" in name:
                # Semi-supervised OT: pass labeled target separately (will be used in transport)
                acc = method(Xs_src, ys_src,
                             Xt_labeled, yt_labeled,
                             Xt_val, yt_val,
                             Xt_test, yt_test)
            else:
                # Other methods: add labeled target to source training set
                Xs_aug = np.vstack([Xs_src, Xt_labeled])
                ys_aug = np.concatenate([ys_src, yt_labeled])
                acc = method(Xs_aug, ys_aug,
                             Xt_val, yt_val,
                             Xt_test, yt_test)

            scores[name] += acc
            print(f"  => Test accuracy this repetition: {acc:.4f}\n")

    # Average
    for name in scores:
        scores[name] /= n_repetitions
    return scores







# ----------------------------------------------------------------------
# 0. Utility functions (assumed to be defined elsewhere)
# ----------------------------------------------------------------------
# Your dataset classes: UspsDataset, MnistDataset, PieDataset, CaltechOfficeDataset
# They are assumed to exist and work correctly.
# Signatures kept for code consistency.

# ----------------------------------------------------------------------
# 1. JDA function already defined elsewhere, just imported here
#    (method_jda is assumed to be available)
# ----------------------------------------------------------------------
# from your_module import method_jda   # import your existing JDA

# ----------------------------------------------------------------------
# 2. Existing methods (baseline, PCA, GFK, TSL, OT...)
#    They are assumed to be defined. For TSL, we keep the pipeline version.
# ----------------------------------------------------------------------
# (Functions method_baseline_1nn, method_pca, method_gfk, method_tsl, method_ot_* are unchanged.)
# Note: method_tsl now returns (acc, best_k, best_mu).

# ----------------------------------------------------------------------
# 3. Function to run a single pair and return the average accuracy
# ----------------------------------------------------------------------
def run_single_pair(load_source, load_target, method_fn, n_repetitions=10, seed=0):
    """Run n_repetitions and return the mean accuracy."""
    np.random.seed(seed)
    accs = []
    for _ in range(n_repetitions):
        Xs, ys = load_source(seed)
        Xt, yt = load_target(seed)
        # Split target into validation and test (50/50)
        n = len(Xt)
        idx = np.random.permutation(n)
        n_val = n // 2
        Xt_val, yt_val = Xt[idx[:n_val]], yt[idx[:n_val]]
        Xt_test, yt_test = Xt[idx[n_val:]], yt[idx[n_val:]]
        # Run the method
        res = method_fn(Xs, ys, Xt_val, yt_val, Xt_test, yt_test)
        if isinstance(res, tuple):
            acc = res[0]
        else:
            acc = res
        accs.append(acc)
    valid_accs = [a for a in accs if a is not None]
    return np.mean(valid_accs)

# ----------------------------------------------------------------------
# 4. Create and save a table as an image
# ----------------------------------------------------------------------
def save_table_image(df, title, filename):
    """Save a pandas DataFrame as a PNG image."""
    fig, ax = plt.subplots(figsize=(len(df.columns)*1.2, len(df.index)*0.6))
    ax.axis('off')
    table = ax.table(cellText=df.round(2).values,
                     rowLabels=df.index,
                     colLabels=df.columns,
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.2)
    plt.title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()

# ----------------------------------------------------------------------
# 5. Collect results for a group of pairs
# ----------------------------------------------------------------------
def build_results_table(pairs, methods, n_repetitions, table_title, filename):
    """
    Build a results table.

    Parameters
    ----------
    pairs : list of tuples (src_name, tgt_name, load_src_fn, load_tgt_fn)
    methods : dict {method_name: function}
    n_repetitions : int
    table_title : str
    filename : str
    """
    rows = []
    for src_name, tgt_name, load_src, load_tgt in pairs:
        row = {}
        for method_name, method_fn in methods.items():
            mean_acc = run_single_pair(load_src, load_tgt, method_fn, n_repetitions)
            row[method_name] = mean_acc
        rows.append((f"{src_name}→{tgt_name}", row))
    # Compute mean row
    mean_row = {}
    for method_name in methods.keys():
        vals = [r[1][method_name] for r in rows]
        mean_row[method_name] = np.mean(vals)
    rows.append(("mean", mean_row))
    # DataFrame
    df = pd.DataFrame([r[1] for r in rows], index=[r[0] for r in rows])
    df.index.name = "Domains"
    # Save image
    save_table_image(df, table_title, filename)
    return df

# ----------------------------------------------------------------------
# 6. Main program
# ----------------------------------------------------------------------
if __name__ == "__main__":
    dims = [20, 40, 60]
    regs = [0.01,1, 100]   # enlever 1e-3 et 1000

    # Dictionary of methods for SURF / PIE / MNIST
    methods = {
        "1NN": method_baseline_1nn,
        "PCA": partial(method_pca, dims=dims),
        "GFK": partial(method_gfk, dims=dims),
        "TSL": partial(method_tsl, dims=dims, regs=regs),
        "OT-exact": method_ot_exact,
        "OT-IT": partial(method_ot_entropic, regs=regs),    # entropic OT
        "OT-Laplace": partial(method_ot_laplacian, regs=regs),
        "OT-LpLq": partial(method_ot_lpl1, regs=regs),      # LpL1 OT
        "OT-GL": partial(method_ot_group_lasso, regs=regs),
    }

    # Data loading functions
    def load_usps(seed): return UspsDataset(seed=seed).generate()
    def load_mnist(seed): return MnistDataset(seed=seed).generate()
    def load_pie(seed, pose): return PieDataset(pose)
    def caltech_load(seed, representation, domain):
        return CaltechOfficeDataset(representation=representation, domain=domain)
    

    domains = ["caltech", "amazon", "webcam", "dslr"]

    # MNIST <-> USPS
    pairs_mnist = [("M", "U", load_mnist, load_usps),
                   ("U", "M", load_usps, load_mnist)]
    build_results_table(pairs_mnist, methods, n_repetitions=3,
                        table_title="MNIST-USPS", filename="results_mnist.png")
    
    # ------------------------------------------------------------------
    #  Caltech-Office SURF
    # ------------------------------------------------------------------
    
    pairs_surf = []
    for src in domains:
        for tgt in domains:
            if src != tgt:
                src_fn = partial(caltech_load, representation="SURF", domain=src)
                tgt_fn = partial(caltech_load, representation="SURF", domain=tgt)
                pairs_surf.append((src[0].upper(), tgt[0].upper(), src_fn, tgt_fn))  # e.g., C, A
    build_results_table(pairs_surf, methods, n_repetitions=3,
                        table_title="Office+Caltech SURF", filename="results_surf.png")

    # ------------------------------------------------------------------
    #  PIE
    # ------------------------------------------------------------------
    poses = ["LeftPose", "UpwardPose", "DownwardPose", "RightPose"]
    # Mapping to short names (P1, P2, P3, P4)
    pose_map = {"LeftPose": "P1", "UpwardPose": "P2", "DownwardPose": "P3", "RightPose": "P4"}
    pairs_pie = []
    for p1 in poses:
        for p2 in poses:
            if p1 != p2:
                src_fn = partial(load_pie, pose=p1)
                tgt_fn = partial(load_pie, pose=p2)
                pairs_pie.append((pose_map[p1], pose_map[p2], src_fn, tgt_fn))
    build_results_table(pairs_pie, methods, n_repetitions=3,
                        table_title="PIE", filename="results_pie.png")

    

    # ------------------------------------------------------------------
    #  DeCAF layer
    # ------------------------------------------------------------------
    # Methods for DeCAF: DeCAF (1NN), JDA, OT-IT, OT-GL
    methods_decaf = {
        "DeCAF": method_baseline_1nn,
        #"JDA": partial(method_jda, dims=dims, regs=regs),
        "OT-IT": partial(method_ot_entropic, regs=regs),
        "OT-GL": partial(method_ot_group_lasso, regs=regs),
    }

    # Layer 6
    pairs_decaf6 = []
    for src in domains:
        for tgt in domains:
            if src != tgt:
                src_fn = partial(caltech_load, representation="DeCAF", domain=src)
                tgt_fn = partial(caltech_load, representation="DeCAF", domain=tgt)
                # Assumes DeCAF features are layer 6 by default
                pairs_decaf6.append((src[0].upper(), tgt[0].upper(), src_fn, tgt_fn))
    build_results_table(pairs_decaf6, methods_decaf, n_repetitions=3,
                        table_title="DeCAF Layer 6", filename="results_decaf")



    print("All tables have been saved as PNG files.")




    
    

