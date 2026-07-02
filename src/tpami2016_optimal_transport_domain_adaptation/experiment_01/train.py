from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from transport import TransportedSamples
from tpami2016_optimal_transport_domain_adaptation.datasets import TwoMoonsDataset
from pbda import Pbda
from tpami2016_optimal_transport_domain_adaptation.datasets import Dataset
from sklearn.base import clone
from skada import DASVMClassifier, source_target_split
from skada.datasets import make_dataset_from_moons_distribution  
import numpy as np
from sklearn.metrics import accuracy_score


angles = [90,70,50,40,30,20,10]
for j in range(len(angles)):
    print(f"angle = {angles[j]}")
    print("=====================================")
    scores = np.zeros(3)
    for i in range(10):
            moons_data = TwoMoonsDataset(seed=i)
            Xs, ys = moons_data.source(n_samples=(150, 150))
            Xt,yt = moons_data.target((150,150), angles[j])
            samples = TransportedSamples(seed = i)
            samples.generate()
            X_train,y_train = samples.transported_samples(regularization = "None",angle = angles[j] )
            X_test,y_test = Xt,yt = moons_data.target(1000, angles[j])
            # SVM with a Gaussian kernel, 
            svm = SVC(kernel='rbf', random_state=42)
            param_grid = {
                'C': [0.1, 1, 10, 100, 1000],           # Régularisation
                'gamma': [0.001, 0.01, 0.1, 1, 10, 'scale', 'auto']  # Paramètre du noyau Gaussien
            }

            # 5-fold cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            # GridSearchCV
            grid_search = GridSearchCV(
                estimator=svm,
                param_grid=param_grid,
                cv=cv,
                scoring='accuracy',      
                n_jobs=-1,               
                verbose=1
            )

            model1 = make_pipeline(
                StandardScaler(),        
                grid_search
            )
            model1.fit(X_train,y_train)
            y_pred = model1.predict(X_test)

            scores[0] += accuracy_score(y_test, y_pred)

            # ---------------------------------------------------------
            # DASVM classifier
            # ---------------------------------------------------------
            best_svm = model1.named_steps['gridsearchcv'].best_estimator_
            dasvm = DASVMClassifier(
                base_estimator = best_svm,
                k=5,              # number of target samples added/removed per class per iteration
                max_iter=20,      # number of self-training iterations

                # -----------------------------------------------------
                # Optional debugging / analysis flags
                # -----------------------------------------------------
                save_estimators=True,  # store intermediate SVM models (useful for analysis / plotting convergence)
                save_indices=False     # set True if you want to track which samples were moved between domains
            )

            X = np.vstack([Xs, Xt])  # ou np.concatenate((Xs, Xt), axis=0)
            y = np.concatenate([ys, np.full(len(Xt), -1)])   # -1 pour les target (non labellisées)

            sample_domain = np.concatenate([
                np.ones(len(Xs), dtype=int),   # source = +1
                -np.ones(len(Xt), dtype=int)   # target = -1
            ])

            # 2. Entraîne le modèle
            dasvm = DASVMClassifier(
                base_estimator=clone(svm), 
                k=5, 
                max_iter=1000,
                save_estimators=True,
                save_indices=True
            )
            dasvm.fit(X, y, sample_domain=sample_domain)   
            y_pred_dasvm = dasvm.predict(X_test)
            scores[1] += accuracy_score(y_test, y_pred_dasvm)

            #pbda model 

            pbda = Pbda(A=1.0, C=1.0)
            source_data = Dataset(Xs, ys)
            target_data = Dataset(Xt, yt)
            classifier = pbda.learn(source_data, target_data)
            y_pred_pbda = classifier.predict(X_test)
            y_pred_pbda_class = (y_pred_pbda > 0.5).astype(int)   # ou 0 selon tes classes
            scores[2] += accuracy_score(y_test, y_pred_pbda_class)
    print(scores/10)
