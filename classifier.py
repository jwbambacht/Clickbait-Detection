from dataset import Dataset
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from resample import Resample
from evaluation import Evaluation
import matplotlib.pyplot as plt
import scikitplot as skplt


def run_classifier(name,plt):
    # Create the classifier
    if name == "random_forest":
        clf = RandomForestClassifier()
        j = 1
    elif name == "ada_boost":
        clf = AdaBoostClassifier()
        j = 2
    else:
        raise ValueError("Invalid classifier provided.")

    # Import the small and big dataset.
    small_dataset = Dataset("datasets/small_training")
    big_dataset = Dataset("datasets/big_training")

    # Combine the two datasets
    resampled_features, resampled_labels = Resample.undersample(small_dataset, big_dataset)

    # Split the data into a train and test set
    X_train, y_train = shuffle(resampled_features, resampled_labels, random_state=42)

    # Split the train data into folds
    kf = KFold(n_splits=10)
    i = 1
    
    # Arrays for each metric
    acc = []
    auc = []
    tpr = []
    fpr = []
    f1 = []
    prec = []
    rec = []
    tp = []
    tn = []
    fp = []
    fn = []

    colors = ["","red","blue"]

    for train_index, test_index in kf.split(X_train):

        X_fold_train, X_fold_test = X_train[train_index], X_train[test_index]
        y_fold_train, y_fold_test = y_train[train_index], y_train[test_index]

        clf.fit(X_fold_train, y_fold_train)
        
        truth = y_fold_test
        pred = clf.predict(X_fold_test)
        data = np.vstack((truth,pred)).T
        eval = Evaluation(data)
        
        acc.append(eval.confusion_matrix_accuracy)
        auc.append(eval.auc_score)
        tpr.append(eval.true_positive_rate[1])
        fpr.append(eval.false_positive_rate[1])
        f1.append(eval.f1_score)
        prec.append(eval.f1_precision)
        rec.append(eval.f1_recall)
        
        tp.append(eval.confusion_matrix[1][1])
        tn.append(eval.confusion_matrix[0][0])
        fp.append(eval.confusion_matrix[0][1])
        fn.append(eval.confusion_matrix[1][0])

        i += 1
    
    print(f"----------------------------------------------------------------------------------")
    print(f"---------------------------------- Mean of {name} --------------------------------")
    print(f"Accuracy: {np.mean(acc)}")
    print(f"AUC: {np.mean(auc)}")
    print(f"TPR: {np.mean(tpr)}")
    print(f"FPR: {np.mean(fpr)}")
    print(f"F1: {np.mean(f1)}")
    print(f"Precision: {np.mean(prec)}")
    print(f"Recall: {np.mean(rec)}")
    
    print(f"TP: {np.mean(tp)}")
    print(f"TN: {np.mean(tn)}")
    print(f"FP: {np.mean(fp)}")
    print(f"FN: {np.mean(fn)}")
    
    roc_label = '{} (AUC={:.2f})'.format(str(name), np.mean(auc))
    plt.plot([0, np.mean(fpr), 1], [0, np.mean(tpr), 1], color=colors[j], linewidth=2, label=roc_label)

    return plt


