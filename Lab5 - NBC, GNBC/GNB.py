import numpy as np
import operator
from scipy.stats import norm

class GaussianNaiveBayesianClassifier :


    def __init__(self, epsilon = 0.01):
        self.__label_distr = {}
        self.__epsilon = epsilon
        
    def fit(self, train_features, train_labels):
        for unique_label in np.unique(train_labels):
            label_indices = np.where(train_labels==unique_label)
            selected_samples = train_features[label_indices]
            mean = np.mean(selected_samples, axis=0)
            std = np.std(selected_samples, axis=0)
            distr_obj = [norm(mean[i], std[i]+self.__epsilon) for i in range(len(mean))]
            self.__label_distr.update({unique_label:distr_obj})
            
            
    def predict(self, test_features):
        pred = np.zeros(test_features.shape[0])
        for i, feature_sample in enumerate(test_features):
            label_prob = {}
            for label, label_distr_obj in self.__label_distr.items():
                prob = 1
                for j in range(feature_sample.size):
                    prob *= label_distr_obj[j].pdf(feature_sample[j])
                label_prob.update({label:prob})
            pred[i] = max(label_prob.items(), key=operator.itemgetter(1))[0]
        return pred
            