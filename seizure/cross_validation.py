'''
Created on Aug 30, 2014

@author: Michael
'''
import numpy
from sklearn import cross_validation

def split_train_random(X, y, cv_ratio):
    X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(X, y, test_size=cv_ratio, random_state=0)
    return X_train, y_train, X_cv, y_cv

def split_train_ictal(X, y, latencies, cv_ratio):
    seizure_ranges = seizure_ranges_for_latencies(latencies)
    seizure_durations = [r[1] - r[0] for r in seizure_ranges]

    num_seizures = len(seizure_ranges)
    num_cv_seizures = int(max(1.0, num_seizures * cv_ratio))

    # sort seizures by biggest duration first, then take the middle chunk for cross-validation
    # and take the left and right chunks for training
    tagged_durations = zip(range(len(seizure_durations)), seizure_durations)
    tagged_durations.sort(cmp=lambda x,y: cmp(y[1], x[1]))
    middle = num_seizures / 2
    half_cv_seizures = num_cv_seizures / 2
    start = middle - half_cv_seizures
    end = start + num_cv_seizures

    chosen = tagged_durations[start:end]
    chosen.sort(cmp=lambda x,y: cmp(x[0], y[0]))
    cv_ranges = [seizure_ranges[r[0]] for r in chosen]

    train_ranges = []
    prev_end = 0
    for start, end in cv_ranges:
        train_start = prev_end
        train_end = start

        if train_start != train_end:
            train_ranges.append((train_start, train_end))

        prev_end = end

    train_start = prev_end
    train_end = len(latencies)
    if train_start != train_end:
        train_ranges.append((train_start, train_end))

    X_train_chunks = [X[start:end] for start, end in train_ranges]
    y_train_chunks = [y[start:end] for start, end in train_ranges]

    X_cv_chunks = [X[start:end] for start, end in cv_ranges]
    y_cv_chunks = [y[start:end] for start, end in cv_ranges]

    X_train = numpy.concatenate(X_train_chunks)
    y_train = numpy.concatenate(y_train_chunks)
    X_cv = numpy.concatenate(X_cv_chunks)
    y_cv = numpy.concatenate(y_cv_chunks)

    return X_train, y_train, X_cv, y_cv

def seizure_ranges_for_latencies(latencies):
    indices = numpy.where(latencies == 0)[0]

    ranges = []
    for i in range(1, len(indices)):
        ranges.append((indices[i-1], indices[i]))
    ranges.append((indices[-1], len(latencies)))

    return ranges