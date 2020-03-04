from dataset import Dataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from resample import Resample

# Import the small and big dataset.
small_dataset = Dataset("datasets/small_training")
big_dataset = Dataset("datasets/big_training")

# Combine the two datasets
resampled_features, resampled_labels = Resample.undersample(small_dataset, big_dataset)

# Split the data into a train and test set
X_train, X_test, y_train, y_test = train_test_split(resampled_features, resampled_labels, test_size=0.1, shuffle=True)

# Create the classifier
clf = RandomForestClassifier()

# Split the train data into folds
kf = KFold(n_splits=10)
for train_index, test_index in kf.split(X_train):
    X_fold_train, X_fold_test = X_train[train_index], X_train[test_index]
    y_fold_train, y_fold_test = y_train[train_index], y_train[test_index]

    clf.fit(X_fold_train, y_fold_train)
    print(clf.score(X_fold_test, y_fold_test))

# Test the classifier on the test set
test_accuracy = clf.score(X_test, y_test)
print('Final test set score: ' + str(test_accuracy))



