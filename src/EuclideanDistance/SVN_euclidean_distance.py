from sklearn import svm
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

# Data
X, y = make_blobs(n_samples=100, centers=2, random_state=42)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# SVM linear
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)

print("Hyperplane coefficients:", clf.coef_)
