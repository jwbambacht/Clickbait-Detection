import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score

class Evaluation:
    
    def __init__(self,data):
        self.data = data
        self.sample_size = len(data)
        self.confusion_matrix = [[0]*2 for i in [0]*2]
        
        self.confusion_matrix_accuracy = 0
        self.f1_precision = 0
        self.f1_recall = 0
        self.f1_score = 0
        self.auc_score = 0
        self.false_positive_rate = []
        self.true_positive_rate = []
        
        self._confusion_matrix()
        self._confusion_matrix_accuracy()
        self._area_under_curve()
        self._f1()
        
    def _confusion_matrix(self):
        
        ## Truth \ Predicted
        ## No Clickbait  / Clickbait
        ## TrueNegative  | FalsePositive
        ## FalseNegative | TruePositive
        
        for i in range(len(self.data)):
            if(self.isTruePositive(self.data[i])):
                self.confusion_matrix[1][1] += 1
            elif(self.isTrueNegative(self.data[i])):
                self.confusion_matrix[0][0] += 1
            elif(self.isFalsePositive(self.data[i])):
                self.confusion_matrix[0][1] += 1
            elif(self.isFalseNegative(self.data[i])):
                self.confusion_matrix[1][0] += 1
    
    def _confusion_matrix_accuracy(self):
        self.confusion_matrix_accuracy = (self._get_true_positives()+self._get_true_negatives())/self.sample_size
        
    def _area_under_curve(self):
        self.false_positive_rate, self.true_positive_rate, thresholds = roc_curve(self.data[:,0],self.data[:,1])
        self.auc_score = auc(self.false_positive_rate, self.true_positive_rate)

    def _f1(self):
        self.f1_precision = self._get_true_positives()/(self._get_true_positives()+self._get_false_positives())
        self.f1_recall = self._get_true_positives()/(self._get_true_positives()+self._get_false_negatives())
        self.f1_score = 2 * 1/(1/self.f1_precision+1/self.f1_recall)
    
    @staticmethod
    def isTruePositive(item):
        if(item[0] == item[1] and item[0] == 1):
            return True
        
    @staticmethod
    def isTrueNegative(item):
        if(item[0] == item[1] and item[0] == 0):
            return True
    
    @staticmethod
    def isFalsePositive(item):
        if(item[0] == 0 and item[1] == 1):
            return True
    
    @staticmethod
    def isFalseNegative(item):
        if(item[0] == 1 and item[1] == 0):
            return True
        
    def _get_true_positives(self):
        return self.confusion_matrix[1][1]
    
    def _get_true_negatives(self):
        return self.confusion_matrix[0][0]
    
    def _get_false_negatives(self):
        return self.confusion_matrix[1][0]
    
    def _get_false_positives(self):
        return self.confusion_matrix[0][1]