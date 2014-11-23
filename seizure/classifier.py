'''
Created on Sep 5, 2014

@author: newuser
'''
import numpy
from scipy.ndimage import convolve
from sklearn import cross_validation, preprocessing
from sklearn import svm
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import RFECV, SelectKBest, VarianceThreshold
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, mutual_info_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import BernoulliRBM
from sklearn.svm import SVR

class Processor(object):
    """
    Represents a class that processes features using sklearn
    """
    def __init__(self, name, clf):
        self.name = name
        self.clf = clf
        
    def get_name(self):
        return self.name
    
    def fit(self, X, Y):
        return self.clf.fit(X, Y)
    
def list_to_mat(features_1, features_2):
        features_1 = numpy.array(features_1)
        features_2 = numpy.array(features_2)
        X = numpy.concatenate([features_1,features_2], axis=0)
        X = preprocessing.scale(X)
        Y = numpy.append(numpy.zeros([len(features_1),1]),
          numpy.ones([len(features_2),1]))
        return X, Y

class Classifier(Processor):
    """
    Linear classifier that performs supervised learning on the clips and predicts
    the preictal probability.
    """
    def __init__(self, name, clf):
        Processor.__init__(self, name, clf)
        
    def train(self, interictal_features, preictal_features):
        X, Y = list_to_mat(interictal_features, preictal_features)
        return self.fit(X, Y)
        
    def classify(self, test_features):
        X = numpy.array(test_features)
        return self.clf.predict(X)
               
    def predict(self, test_features):
        X = numpy.array(test_features)
        return self.guess_probability(X)
    
    def cross_validate(self, interictal_features, preictal_features, splits, scoring="roc_auc"):
        X, Y = list_to_mat(interictal_features, preictal_features)
        skf = cross_validation.StratifiedKFold(Y, splits)
        return cross_validation.cross_val_score(self.clf, X, Y, scoring=scoring, cv=skf)
        
    def guess_probability(self, X):
        probs = self.clf.predict_proba(X)
        return probs[:,1]
        
    def auc(self, test_features, y_true):
        y_scores = self.predict(test_features)
        return roc_auc_score(y_true, y_scores)
    
class SVM(Classifier):
    
    def __init__(self, **params):
        # By default use probabilistic classifier
        param_str = ",".join(["%s-%s" % (name, str(value)) for name, value in params.iteritems()])
        param_str = "," + param_str if param_str != "" else param_str
        if "probability" not in params:
            params["probability"] = True
        clf = svm.SVC(**params)
        Classifier.__init__(self, "SVM" + param_str, clf)

class RandomForest(Classifier):
    
    def __init__(self, **params):
        param_str = ",".join(["%s-%s" % (name, str(value)) for name, value in params.iteritems()])
        param_str = "," + param_str if param_str != "" else param_str
        clf = RandomForestClassifier(**params)
        Classifier.__init__(self, "Random Forest" + param_str, clf)
    
    def feature_importance(self):
        return self.clf.feature_importances_
    
class DecisionTree(Classifier):
    def __init__(self, **params):
        param_str = ",".join(["%s-%s" % (name, str(value)) for name, value in params.iteritems()])
        param_str = "," + param_str if param_str != "" else param_str
        clf = DecisionTreeClassifier(**params)
        Classifier.__init__(self, "Decision Tree" + param_str, clf)
        
class ClassifierPipeline(Classifier):
    def __init__(self, classifier, feature_selector):
        
        clf = Pipeline([
          ('feature_selection', feature_selector.clf),
          ('classification', classifier.clf)
        ])
        
        Classifier.__init__(self, "Pipeline-%s-%s" % (classifier.get_name(), feature_selector.get_name()), clf)

class ExtraTrees(Classifier):
    def __init__(self):
        clf = ExtraTreesClassifier()
        Classifier.__init__(self, "ExtraTrees", clf)
        
    def feature_importance(self):
        return self.clf.feature_importances_

class FeatureSelector(Processor):
    """
    Feature selector that chooses best features
    """
    def __init__(self, name, selector):
        Processor.__init__(self, name, selector)
        
    def select(self, interictal_features, preictal_features):
        X, Y = list_to_mat(interictal_features, preictal_features)
        return self.fit(X, Y)
    
class UnivariateSelector(FeatureSelector):
    pass
    
class VarianceSelector(FeatureSelector):
    def __init__(self, **params):
        selector = VarianceThreshold(**params)
        param_str = ",".join(["%s-%s" % (name, str(value)) for name, value in params.iteritems()])
        param_str = "," + param_str if param_str != "" else param_str
        FeatureSelector.__init__(self, "VarianceSelector" + param_str, selector)
        
    def variance(self):
        return self.clf.variances_
    
class KBestSelector(FeatureSelector):
    def __init__(self, **params):
        param_str = ",".join(["%s-%s" % (name, str(value)) for name, value in params.iteritems()])
        param_str = "," + param_str if param_str != "" else param_str
        #if "score_func" not in params:
        #    params["score_func"] = roc_auc_score
        selector = SelectKBest(**params)
        FeatureSelector.__init__(self, "KBestSelector" + param_str, selector)
        
class RFECVSelector(FeatureSelector):
    def __init__(self, **params):
        selector = RFECV(SVR(kernel="linear"), **params)
        FeatureSelector.__init__(self, "RFECVSelector", selector)
        
class NeuralNetwork(FeatureSelector):
    def __init__(self, **params):
        param_str = ",".join(["%s-%s" % (name, str(value)) for name, value in params.iteritems()])
        param_str = "," + param_str if param_str != "" else param_str
        selector = BernoulliRBM(**params)
        FeatureSelector.__init__(self, "BernoulliRBM" + param_str, selector)
        
    def select(self, interictal_features, preictal_features):
        X, Y = list_to_mat(interictal_features, preictal_features)
        return self.fit(X, Y)
        
    def fit(self, X, Y):
        X, Y = nudge_dataset(X, Y)
        return self.clf.fit(X, Y)
    
def nudge_dataset(X, Y):
    """
    This produces a dataset 5 times bigger than the original one,
    by moving the images by 1 pixel
    """
    direction_vectors = [
        [[0, 1, 0],
         [0, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [1, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 1],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 0],
         [0, 1, 0]]]

    shift = lambda x, w: convolve(x.reshape((8, 8)), mode='constant',
                                  weights=w).ravel()
    X = numpy.concatenate([X] +
                       [numpy.apply_along_axis(shift, 1, X, vector)
                        for vector in direction_vectors])
    Y = numpy.concatenate([Y for _ in range(5)], axis=0)
    return X, Y
