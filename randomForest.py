from dataset import Dataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier

# Import the small and big dataset.
small_dataset = Dataset("datasets/small_training")
big_dataset = Dataset("datasets/big_training")

# Combine the two datasets
all_features = np.r_[small_dataset.get_features(), big_dataset.get_features()]
all_labels = np.r_[small_dataset.get_target_labels(), big_dataset.get_target_labels()]

# Resample (undersampling)
combined = np.c_[all_features, all_labels]
no_clickbait = combined[combined[:, -1] == 0]
clickbait = combined[combined[:, -1] == 1]
nr_of_samples = np.size(clickbait, axis=0)
ids = np.random.randint(np.size(no_clickbait, axis=0), size=nr_of_samples)
no_clickbait_resampled = no_clickbait[ids, :]

resampled_features = np.r_[clickbait[:, :-1], no_clickbait_resampled[:, :-1]]
resampled_labels = np.r_[clickbait[:, -1], no_clickbait_resampled[:, -1]]

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



