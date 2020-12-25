import numpy as np
import operator

class NaiveBayesianClassifier:


    def __init__(self):
        self.__label_prob = {}
        self.__feature_val_prob = {}
        
    def fit(self, train_features, train_labels):
        total_samples = train_features.shape[0]
        for unique_label, count in zip(*np.unique(train_labels, return_counts=True)):
            self.__label_prob.update({unique_label:count/total_samples})
            
            label_indices = np.where(train_labels==unique_label)
            selected_samples = train_features[label_indices]
            samples = selected_samples.shape[0]
            feature_val_list = []
            for feature in selected_samples.T:
                pixel_value, pixel_value_count = np.unique(feature, return_counts=True)
                value_counts = dict(zip(pixel_value, pixel_value_count))
                value_counts = {k:v/samples for (k,v) in value_counts.items()}
                feature_val_list.append(value_counts)
            
            self.__feature_val_prob.update({unique_label:feature_val_list})
            
    def predict(self, test_features):
        pred = np.zeros(test_features.shape[0])
        for i, feature_sample in enumerate(test_features):
            seq_prob = {}
            for label, label_prob in self.__label_prob.items():
                prob = label_prob
                feature_val_list = self.__feature_val_prob[label]
                for j, feature_val in enumerate(feature_sample):
                    if feature_val in feature_val_list[j]:
                        prob *= feature_val_list[j][feature_val]
                    else:
                        prob *= 0.0001
                        #break
                seq_prob.update({label:prob})
            pred[i] = max(seq_prob.items(), key=operator.itemgetter(1))[0]
        return pred
            

