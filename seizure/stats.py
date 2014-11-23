'''
Created on Aug 31, 2014

@author: Michael
'''
import numpy
from scipy import stats

from seizure.classifier import list_to_mat

def statistical_analyze(interictal, preictal):
    results = []
    feature_indices = interictal[0].get_feature_indices()
    interictal_array = numpy.array(interictal)
    preictal_array = numpy.array(preictal)
    
    for feature, indices in feature_indices.iteritems():
        first, last = indices
        interictal_part = interictal_array[first:last]
        preictal_part = preictal_array[first:last]
        interictal_mean = numpy.mean(interictal_part, axis=0)
        preictal_mean = numpy.mean(preictal_part, axis=0)
        interictal_std = numpy.std(interictal_part, axis=0)
        preictal_std = numpy.std(interictal_part, axis=0)
        mean_diff = interictal_mean - preictal_mean
        pvalues = []
        for i, column in enumerate(interictal_part.T):
            __, pvalue = stats.ttest_ind(column, preictal_part[:,i])
            pvalues.append(pvalue)
        pvalues = numpy.array(pvalues)
        results.append([("Feature", feature), ("Mean-Interictal", interictal_mean), ("Median-Interictal", numpy.median(interictal_part, axis=0)),
      ("Std-Interictal", interictal_std), ("Mean-Preictal", preictal_mean),
      ("Median-Preictal", numpy.median(preictal_part, axis=0)), ("Std-Preictal", numpy.std(preictal_part, axis=0)),
      ("Mean-Diff", mean_diff), ("Confidence Interval", preictal_std), ("T-test pvalues", pvalues)])
        
    return results

def mutual_information(X, Y):
    pass

def correlate(interictal,preictal):
    X, Y = list_to_mat(interictal,preictal)
    output = numpy.zeros([len(X.T),1])
    for i, column in enumerate(X.T):
        output[i] = numpy.corrcoef(column,Y)
    return output
    