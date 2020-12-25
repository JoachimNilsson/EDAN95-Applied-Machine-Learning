import numpy as np
import operator

class NearestCentroidClassifier:


    def __init__(self):
        self.__centroids = {}
        
    def fit(self, train_features, train_labels):
        for unique_label in np.unique(train_labels):
            label_indices = np.where(train_labels==unique_label)
            selected_samples = train_features[label_indices]
            centroid = np.mean(selected_samples, axis=0)
            self.__centroids.update({unique_label:centroid})
            
    def predict(self, test_features):
        pred = np.zeros(test_features.shape[0])
        for i, feature_sample in enumerate(test_features):
            pred[i] = sorted(self.__centroids.items(), key = lambda i: np.linalg.norm(i[1]-feature_sample))[0][0]
            #pred[i] = max(self.__centroids.items(), key = np.linalg.norm(operator.itemgetter(1)-feature_sample))
        return pred
            