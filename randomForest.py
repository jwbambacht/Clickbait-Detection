from dataset import Dataset
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from resample import Resample
from evaluation import Evaluation
import matplotlib.pyplot as plt

# Import the small and big dataset.
small_dataset = Dataset("datasets/small_training")
big_dataset = Dataset("datasets/big_training")

# Combine the two datasets
resampled_features, resampled_labels = Resample.undersample(small_dataset, big_dataset)



# Split the data into a train and test set
X_train, y_train = shuffle(resampled_features, resampled_labels, random_state=42)

# Create the classifier
clf = RandomForestClassifier()

# Split the train data into folds
kf = KFold(n_splits=10)
i = 1

plt.figure(figsize=(8, 8))
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curve")

x = [0.0, 1.0]
plt.plot(x, x, linestyle='dashed', color='black', linewidth=2, label='Random')

colors = ["","red","blue","yellow","green","black","gray","magenta","cyan","turquoise","gold"]

for train_index, test_index in kf.split(X_train):

    X_fold_train, X_fold_test = X_train[train_index], X_train[test_index]
    y_fold_train, y_fold_test = y_train[train_index], y_train[test_index]

    clf.fit(X_fold_train, y_fold_train)

    truth = y_fold_test
    pred = clf.predict(X_fold_test)
    data = np.vstack((truth,pred)).T
    eval = Evaluation(data)

    print(f"----------------------------------------------------------------------------------")
    print(f"---------------------------------- Train set {i} ---------------------------------")
    print(f"Confusion Matrix: {eval.confusion_matrix}, Accuracy: {eval.confusion_matrix_accuracy}")
    print(f"AUC score: {eval.auc_score}, TPR: {eval.true_positive_rate[1]}, FPR: {eval.false_positive_rate[1]}")
    print(f"F1: {eval.f1_score}, Precision: {eval.f1_precision}, Recall: {eval.f1_recall}")
    
    roc_label = '{} (AUC={:.2f})'.format("Train set "+str(i), eval.auc_score)
    plt.plot(eval.false_positive_rate, eval.true_positive_rate, color=colors[i], linewidth=2, label=roc_label)

    i += 1

"""
truth = y_test
pred = clf.predict(X_test)
data = np.vstack((truth,pred)).T
eval = Evaluation(data)
            
print(f"----------------------------------------------------------------------------------")
print(f"------------------------------------ Test set -----------------------------------")
print(f"Confusion Matrix: {eval.confusion_matrix}, Accuracy: {eval.confusion_matrix_accuracy}")
print(f"AUC score: {eval.auc_score}, TPR: {eval.true_positive_rate[1]}, FPR: {eval.false_positive_rate[1]}")
print(f"F1: {eval.f1_score}, Precision: {eval.f1_precision}, Recall: {eval.f1_recall}")

roc_label = '{} (AUC={:.2f})'.format("Test set", eval.auc_score)
plt.plot(eval.false_positive_rate, eval.true_positive_rate, linestyle="dashed", color="red", linewidth=2, label=roc_label)

plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.legend(fontsize=12, loc='lower right')
plt.tight_layout()
plt.savefig("auc-roc")
"""