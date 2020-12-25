from scipy.stats import multivariate_normal
import numpy as np

class ExpectationMaximizationGNB:
    
    def __init__(self, K, epsilon = 0.001, smoothing_factor = 0.01):
        self.__K = K
        self.__epsilon = epsilon
        self.__result = None
        self.__smoothing_factor = smoothing_factor
        
    def fit(self, features):
        n_samples = features.shape[0]
        n_features = features.shape[1]
        initial_class_labels = np.random.randint(0, high=self.__K, size=n_samples)
        old_class_prior = np.zeros(self.__K)
        old_class_mean = np.zeros((n_features, self.__K))
        old_class_cov = np.zeros((n_features, self.__K))
        for i, unique_label in enumerate(np.asarray(np.unique(initial_class_labels, return_counts=True)).T):
            label_indices = np.where(initial_class_labels==unique_label[0])
            selected_samples = features[label_indices]
            mean = np.mean(selected_samples, axis=0)
            var = np.var(selected_samples, axis=0)
            old_class_mean[:,i] = mean
            old_class_cov[:,i] = var
            old_class_prior[i] = unique_label[1]/n_samples

        posterior_prob = self.e_step(features, old_class_prior, old_class_mean, old_class_cov)
        new_class_prior, new_class_mean, new_class_cov = self.m_step(posterior_prob, features)
        while(any(diff>self.__epsilon for diff in np.linalg.norm(new_class_mean-old_class_mean, axis=0))):
            posterior_prob = self.e_step(features, new_class_prior, new_class_mean, new_class_cov)
            old_class_prior, old_class_mean, old_class_cov = new_class_prior, new_class_mean, new_class_cov
            new_class_prior, new_class_mean, new_class_cov = self.m_step(posterior_prob, features)
        self.__result = {'class_prior': new_class_prior, 'class_mean': new_class_mean, 'class_cov':new_class_cov}
        
    def e_step(self, features, class_prior, class_mean, class_cov):
        n_classes = class_prior.shape[0]
        n_samples = features.shape[0]
        n_features = features.shape[1]
        posterior_prob = np.zeros((n_samples, n_classes))
        for sample_ind, feature_sample in enumerate(features):
            for class_ind in range(n_classes):
                norm_dist = multivariate_normal(mean=class_mean[:,class_ind], cov=np.diag(class_cov[:,class_ind])+self.__smoothing_factor*np.eye(n_features), allow_singular=True); 
                posterior_prob[sample_ind, class_ind] = class_prior[class_ind] * norm_dist.pdf(feature_sample)
            
            posterior_prob[sample_ind,:] = posterior_prob[sample_ind,:]/np.sum(posterior_prob[sample_ind,:])
        return posterior_prob
    
    def m_step(self, posterior_prob, features):
        n_samples = features.shape[0]
        n_features = features.shape[1]
        n_classes = self.__K
        posterior_prob_classes = np.sum(posterior_prob, axis=0)
        class_prior = posterior_prob_classes/n_samples
        class_mean = np.zeros((n_features, n_classes))
        class_cov = np.zeros((n_features, n_classes))
        for i, prob_classes in enumerate(posterior_prob_classes):
            class_mean[:,i] = features.T@posterior_prob[:,i]/posterior_prob_classes[i]
            cov_matrix = features.T@np.diag(posterior_prob[:,i])@features/posterior_prob_classes[i] - np.outer(class_mean[:,i],class_mean[:,i])

            class_cov[:,i] = np.diag(cov_matrix)

        return class_prior, class_mean, class_cov
    
    def predict(self, test_features):
        n_features = test_features.shape[1]
        class_prior = self.__result['class_prior']
        class_mean = self.__result['class_mean']
        class_cov = self.__result['class_cov']
        pred = np.zeros(test_features.shape[0])
        for sample_ind, feature_sample in enumerate(test_features):
            class_prob = np.zeros(self.__K)
            for class_ind in range(self.__K):
                norm_dist = multivariate_normal(mean=class_mean[:,class_ind], cov=np.diag(class_cov[:,class_ind])+self.__smoothing_factor*np.eye(n_features), allow_singular=True); 
                class_prob[class_ind] = class_prior[class_ind] * norm_dist.pdf(feature_sample)

            pred[sample_ind] = np.argmax(class_prob)
        return pred
        
        