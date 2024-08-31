import matplotlib.font_manager
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
import joblib


X_train = np.load('train_normal_features.npy')

clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
clf.fit(X_train)

# Save the trained model
joblib.dump(clf, 'one_class_svm_model.pkl')

# # Load features for each slide and make predictions
# for slide_number in ["01", "03", "05", "07", "37", "53", "55", "86", "88", "96", "98", "101"]:
#     # Load features
#     features_path = f'test_slide_malignant_{slide_number}_features.npy'
#     X_test_slide = np.load(features_path)

#     # Predictions
#     y_pred_slide = clf.predict(X_test_slide)
    
#     # Count the number of 1s (normal) and -1s (abnormal)
#     num_normals = np.sum(y_pred_slide == 1)
#     num_abnormals = np.sum(y_pred_slide == -1)
    
#     # Report per-slide statistics
#     total_samples = len(y_pred_slide)
#     print(f"Slide {slide_number}:")
#     print(f"  Number of normals detected: {num_normals}")
#     print(f"  Number of abnormals detected: {num_abnormals}")
#     print(f"  Total number of samples: {total_samples}")

clf = joblib.load('one_class_svm_model.pkl')

# Process each test slide normal dataset
for slide_number in ["78", "80"]:
    # Load features
    features_path = f'test_slide_normal_{slide_number}_features.npy'
    X_test_slide = np.load(features_path)

    # Predictions
    y_pred_slide = clf.predict(X_test_slide)
    
    # Count the number of 1s (normal) and -1s (abnormal)
    num_normals = np.sum(y_pred_slide == 1)
    num_abnormals = np.sum(y_pred_slide == -1)
    
    # Report per-slide statistics
    total_samples = len(y_pred_slide)
    print(f"Slide {slide_number}:")
    print(f"  Number of normals detected: {num_normals}")
    print(f"  Number of abnormals detected: {num_abnormals}")
    print(f"  Total number of samples: {total_samples}")






