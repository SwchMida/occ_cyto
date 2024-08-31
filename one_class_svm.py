import matplotlib.font_manager
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.metrics import f1_score

# Load data
X_train = np.load('train_normal_centercropresize.npy')
X_test = np.load('test_normal_centercropresize.npy')
X_outliers = np.load('test_abnormal_centercropresize_OTH.npy')

# Train One-Class SVM model
clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
clf.fit(X_train)

# Predictions
svm_scores_outliers = clf.decision_function(X_outliers)
y_pred_outliers = clf.predict(X_outliers)

# Sort anomalies by SVM scores
sorted_indices = np.argsort(svm_scores_outliers)
top_100_anomalies = sorted_indices[:100]

# Ground truth labels for the top 100 anomalies
ground_truth_labels = np.ones_like(top_100_anomalies)





