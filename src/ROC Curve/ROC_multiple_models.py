"""
Example 2: Comparing ROC Curves of Multiple Models
"""

import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score

# Generate dataset
X, y = make_classification(
    n_samples=1500,
    n_features=12,
    n_informative=6,
    random_state=0
)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

# Models
log_reg = LogisticRegression()
rf = RandomForestClassifier(n_estimators=200)

log_reg.fit(X_train, y_train)
rf.fit(X_train, y_train)

# Probabilities
log_prob = log_reg.predict_proba(X_test)[:, 1]
rf_prob = rf.predict_proba(X_test)[:, 1]

# ROC curves
fpr_log, tpr_log, _ = roc_curve(y_test, log_prob)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_prob)

# AUC
auc_log = roc_auc_score(y_test, log_prob)
auc_rf = roc_auc_score(y_test, rf_prob)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(fpr_log, tpr_log, label=f"Logistic Regression (AUC = {auc_log:.3f})")
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {auc_rf:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid(True)
plt.show()
