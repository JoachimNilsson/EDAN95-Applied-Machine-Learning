from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestCentroid
import MNIST
import NCC
import NBC


def main() :
    mnist = MNIST.MNISTData('MNIST_Light/*/*.png')


    train_features, test_features, train_labels, test_labels = mnist.get_data()

    mnist.visualize_random()
    gnb = GaussianNB()
    print(train_labels)
    gnb.fit(train_features, train_labels)
    y_pred = gnb.predict(test_features)

    print("Classification report SKLearn GNB:\n%s\n"
      % (metrics.classification_report(test_labels, y_pred)))
    print("Confusion matrix SKLearn GNB:\n%s" % metrics.confusion_matrix(test_labels, y_pred))

    mnist.visualize_wrong_class(y_pred, 8)
    
    clf = NearestCentroid()
    clf.fit(train_features, train_labels)
    y_pred = clf.predict(test_features)
    print("Classification report SKLearn NCC:\n%s\n"
      % (metrics.classification_report(test_labels, y_pred)))
    print("Confusion matrix SKLearn NCC:\n%s" % metrics.confusion_matrix(test_labels, y_pred))
    
    
    nbc1 = NBC.NaiveBayesianClassifier()
    nbc1.fit(train_features, train_labels)
    y_pred = nbc1.predict(test_features)
    print("Classification report implemented NBC:\n%s\n"
          % (metrics.classification_report(test_labels, y_pred)))
    print("Confusion matrix implemented NBC:\n%s" % metrics.confusion_matrix(test_labels, y_pred))
    
    
    

if __name__ == "__main__": main()