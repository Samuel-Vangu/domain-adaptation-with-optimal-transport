from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from transport import TransportedSamples
from tpami2016_optimal_transport_domain_adaptation.datasets import TwoMoonsDataset
from sklearn.base import clone
from skada import DASVMClassifier, source_target_split
from skada.datasets import make_dataset_from_moons_distribution  
import numpy as np
from sklearn.metrics import accuracy_score



angles = [10, 20, 30, 40, 50, 70, 90]
ot_methods = ["exact", "entropic", "group lasso", "laplacian"]

# rows : methods
# cols : angles
results = np.zeros((6, len(angles)))

method_names = [
    "SVM",
    "OT-exact",
    "OT-entropic",
    "OT-group lasso",
    "OT-laplacian",
    "DASVM",
]

for j, angle in enumerate(angles):

    print("\n")
    print("=" * 70)
    print(f"Angle {angle}° ({j+1}/{len(angles)})")
    print("=" * 70)

    scores = np.zeros(6)

    for seed in range(10):

        print(f"   Seed {seed+1}/10")

        #####################################################
        # DATA
        #####################################################

        moons_data = TwoMoonsDataset(seed=seed)

        Xs, ys = moons_data.source(n_samples=(150, 150))
        Xt, yt = moons_data.target(n_samples=(150, 150), angle=angle)

        X_test, y_test = moons_data.target(
            n_samples=1000,
            angle=angle
        )

        samples = TransportedSamples(seed=seed)
        samples.generate()

        #####################################################
        # Grid Search
        #####################################################

        svm = SVC(kernel="rbf", random_state=42)

        param_grid = {
            "C": [0.1, 1, 10, 100, 1000],
            "gamma": [0.001, 0.01, 0.1, 1, 10, "scale", "auto"],
        }

        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=42,
        )

        grid_search = GridSearchCV(
            estimator=svm,
            param_grid=param_grid,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
            verbose=0,
        )

        model = make_pipeline(
            StandardScaler(),
            grid_search,
        )

        #####################################################
        # 1) Standard SVM
        #####################################################

        model.fit(Xs, ys)

        y_pred = model.predict(X_test)

        scores[0] += accuracy_score(y_test, y_pred)

        #####################################################
        # 2) OT + SVM
        #####################################################

        for k, method in enumerate(ot_methods):

            X_train, y_train = samples.transported_samples(
                regularization=method,
                angle=angle,
            )

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            scores[k + 1] += accuracy_score(y_test, y_pred)

        #####################################################
        # Best SVM found by GridSearch
        #####################################################

        best_svm = clone(
            model.named_steps["gridsearchcv"].best_estimator_
        )

        #####################################################
        # 3) DASVM
        #####################################################

        X = np.vstack([Xs, Xt])

        y = np.concatenate([
            ys,
            np.full(len(Xt), -1)
        ])

        sample_domain = np.concatenate([
            np.ones(len(Xs), dtype=int),
            -np.ones(len(Xt), dtype=int)
        ])

        dasvm = DASVMClassifier(
            base_estimator=best_svm,
            k=5,
            max_iter=1,
            save_estimators=False,
            save_indices=False,
        )

        dasvm.fit(
            X,
            y,
            sample_domain=sample_domain,
        )

        y_pred = dasvm.predict(X_test)

        scores[5] += accuracy_score(y_test, y_pred)



    #########################################################
    # Mean error rate for one angle
    #########################################################

    scores /= 10
    errors = 1.0 - scores

    results[:, j] = errors

    print(f"\nMean error rates for angle {angle}°")
    for name, error in zip(method_names, errors):
        print(f"{name:20s}: {error:.4f}")

#############################################################
# Final table
#############################################################

import pandas as pd
import matplotlib.pyplot as plt

#############################################################
# Final table
#############################################################

columns = [f"{angle}°" for angle in angles]

df = pd.DataFrame(
    results,
    index=method_names,
    columns=columns,
)

# Round values for display
df = df.round(3)

# Create figure
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.axis("off")

table = ax.table(
    cellText=df.values,
    rowLabels=df.index,
    colLabels=df.columns,
    cellLoc="center",
    loc="center",
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.6)

plt.title("Final Results (Mean Error Rate)", fontsize=14, pad=20)

# Save the table
plt.savefig("final_results.png", dpi=300, bbox_inches="tight")

plt.close(fig)

print(df)
print("\nTable saved as 'final_results.png'")