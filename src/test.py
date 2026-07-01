from datasets import TwoMoonsDataset
moons_data = TwoMoonsDataset()
import numpy as np
import ot 
from common import cost_matrix
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Source domain
Xs,ys = moons_data.source(n_samples = (150,150))

#target domains with differents rotaion angle

Xt1,yt1= moons_data.target(n_samples = (150,150),angle = 20)
Xt2,yt2= moons_data.target(n_samples = (150,150),angle = 40)
Xt3,yt3= moons_data.target(n_samples = (150,150),angle = 40)

#Source distribution

ps = (1/Xs.shape[0])* np.ones(Xs.shape[0])

#Target distribution

pt = (1/Xt1.shape[0])* np.ones(Xt1.shape[0])





# Computing the cost matrix between the domains

C1 = cost_matrix(Xs,Xt1)
C2 = cost_matrix(Xs,Xt2)
C3 = cost_matrix(Xs,Xt3)

# Computing the differents transport plans 

# The plans obtained using exact OT

exact_plan1 = ot.emd(ps, pt, C1)
exact_plan2 = ot.emd(ps, pt, C2)
exact_plan3 = ot.emd(ps, pt, C3)

# The plans using Ot with entropic regularization

sinkhorn_plan1 = ot.sinkhorn(ps, pt, C1, reg = 9, method='sinkhorn')
sinkhorn_plan2 = ot.sinkhorn(ps, pt, C2, reg = 9, method='sinkhorn')
sinkhorn_plan3 = ot.sinkhorn(ps, pt, C3, reg = 9, method='sinkhorn')

# The plans using the class labels regularization:group lasso regularization

plan_gl1 =  ot.da.sinkhorn_l1l2_gl(ps,ys, pt,C1,9, eta=0.1)
plan_gl2 =  ot.da.sinkhorn_l1l2_gl(ps,ys, pt,C2,9, eta=0.1)
plan_gl3 =  ot.da.sinkhorn_l1l2_gl(ps,ys, pt,C3,9, eta=0.1)

# The plans using the laplacian regularization:

plan_laplace1 = ot.da.emd_laplace(ps,pt,Xs,Xt1,C1)

plan_laplace2 = ot.da.emd_laplace(ps,pt,Xs,Xt2,C2)

plan_laplace3 = ot.da.emd_laplace(ps,pt,Xs,Xt3,C3)

print(plan_laplace3.shape)







# SVM with a Gaussian kernel
svm = SVC(kernel='rbf', random_state=42)

# 5-fold cross-validation
param_grid = {
    'C': [0.1, 1, 10, 100, 1000],           
    'gamma': [0.001, 0.01, 0.1, 1, 10, 'scale', 'auto']  
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=svm,
    param_grid=param_grid,
    cv=cv,
    scoring='accuracy',      # ou 'f1', 'roc_auc', etc. selon ton problème
    n_jobs=-1,               # utilise tous les cœurs
    verbose=1
)

model = make_pipeline(
    StandardScaler(),        # Important pour SVM
    grid_search
)

# Entraînement
model.fit(X_train, y_train)

# Résultats
print("Meilleurs paramètres :", grid_search.best_params_)
print("Meilleur score CV :", grid_search.best_score_)

# Prédictions
y_pred = model.predict(X_test)

