'''
Created on Sep 20, 2014

@author: Michael
'''
from collections import defaultdict, namedtuple
from copy import deepcopy
import csv
import itertools
import logging
import numpy
import os
import time

from seizure.classifier import ExtraTrees, KBestSelector
from seizure.graph import DependencyGraph
from seizure.preprocessor import preprocess
from seizure.stats import statistical_analyze
from seizure.transform import Transform
from seizure.plot import plot

class TaskManager(DependencyGraph):
    """
    Uses the json file to fetch the data then performs a list of tasks.
    This works the same way as a Preprocessor except that the file context
    is used as input and it runs once.
    """
    def __init__(self, tasks, file_context):
        DependencyGraph.__init__(self, tasks, "taskmanager")
        self.__file_context = file_context
        self.__locked = True #Prevents users from calling apply() method
        
    def apply(self, input_args):
        if self.__locked:
            raise AssertionError("Do not call this method directly! Use run() instead.")
        return DependencyGraph.apply(self, input_args)
    
    def traverse_vertex(self, vertex, data):
        return vertex["transform"].apply([self.__file_context, data])
    
    def get_output(self, results):
        outputs = []
        for transform in self.final_transforms:
            output = results[self.get(transform)]
            outputs.append(output)
        outputs = outputs[0] if len(outputs) == 1 else outputs
        return outputs
        
    def run(self):
        self.__locked = False
        output = self.apply([self.__file_context])
        self.__locked = True
        return output

class Task(Transform):
    """
    Performs a task by specifying dependencies via the requires() function.
    This works like a Transform except that it only runs once and
    must be reset to be run again.
    """
    def __init__(self):
        self.output = []
        self.__finished = False
    
    def apply(self, input_args):
        # We assume that the first input argument is always the file context and the rest consists of extra input.
        file_context = input_args[0]
        extra_args = input_args[1:]
        extra_args = extra_args[0] if len(extra_args) == 1 else extra_args
        if not self.__finished:
            self.run(file_context, extra_args)
            self.__finished = True
        return self.get_output()
        
    def run(self, file_context, input_args):
        raise NotImplementedError("You forgot to override this method.")
        
    def get_output(self):
        return self.output
    
    def is_finished(self):
        return self.__finished
    
    def reset(self):
        self.__finished = False
        self.output = []

TrainingSet = namedtuple("TrainingSet", ["target", "preprocessor", "interictal", "preictal"])

class PreprocessTraining(Task):
    def __init__(self, preprocessors, nsplits=10, cache=True, overwrite=False):
        Task.__init__(self)
        self.preprocessors = preprocessors
        self.cache = cache
        self.overwrite = overwrite
        self.nsplits = nsplits
    
    def run(self, file_context, __):
        self.output = []
        for preprocessor in self.preprocessors:
            for target in file_context.targets:
                name = os.path.basename(target)
                logging.info("Preprocessing training datasets for target %s with preprocessor %s" % (name, preprocessor.get_name()))
                interictal = file_context.get_interictal_segments(target)
                processed_interictal = preprocess(interictal, preprocessor, file_context, nsplits=self.nsplits, cache=self.cache, overwrite=self.overwrite)
                preictal = file_context.get_preictal_segments(target)
                processed_preictal = preprocess(preictal, preprocessor, file_context, nsplits=self.nsplits, cache=self.cache, overwrite=self.overwrite)
                self.output.append(TrainingSet(name,preprocessor,processed_interictal,processed_preictal))

TestSet = namedtuple("TestSet", ["target", "preprocessor", "name", "test"])

class PreprocessTest(Task):
    def __init__(self, preprocessors, nsplits = 10, cache=True, overwrite=False):
        Task.__init__(self)
        self.preprocessors = preprocessors
        self.cache = cache
        self.overwrite = overwrite
        self.nsplits = nsplits
    
    def run(self, file_context, __):
        for preprocessor in self.preprocessors:
            for target in file_context.targets:
                name = os.path.basename(target)
                logging.info("Preprocessing test datasets for target %s with preprocessor %s" % (name, preprocessor.get_name()))
                tests = file_context.get_test_segments(target)
                processed_tests = preprocess(tests, preprocessor, file_context, nsplits=self.nsplits, cache=self.cache, overwrite=self.overwrite)
                self.output.append(TestSet(name, preprocessor, list(itertools.chain(*[[os.path.basename(t)]*self.nsplits for t in tests])), processed_tests))
    
TrainedClassifier = namedtuple("TrainedClassifier", ["target", "preprocessor", "classifier"])
        
class Train(Task):
    def __init__(self, preprocessors, classifiers, cache=False, overwrite=False):
        Task.__init__(self)
        self.preprocessors = preprocessors
        self.classifiers = classifiers
        self.cache = cache
        self.overwrite = overwrite
    
    def requires(self):
        return [PreprocessTraining(self.preprocessors, cache=self.cache, overwrite=self.overwrite)]
    
    def run(self, file_context, training_sets):
        logging.info("Training with %d preprocessor(s) and %d classifier(s)" % (len(self.preprocessors),len(self.classifiers)))
        
        preprocessor_sets = defaultdict(list)
        for training_set in training_sets:
            preprocessor_sets[training_set.preprocessor].append(training_set)
    
        # Test all preprocesser and classifier combinations to compare results
        for classifier in self.classifiers:
            for preprocessor, training_set_partition in preprocessor_sets.iteritems():
                start = time.time()
            
                logging.info('Using preprocessor %s with classifier %s' % (preprocessor.get_name(), classifier.get_name()))
            
                for training_set in training_set_partition:
                    logging.info("Training target %s with classifier %s" % (training_set.target, classifier.get_name()))
                    model = classifier.train(training_set.interictal, training_set.preictal)
                    self.output.append(TrainedClassifier(training_set.target, training_set.preprocessor, deepcopy(classifier)))
                    
                if self.cache:
                    file_context.cache_dump(model, "%s-%s" % (training_set.target, classifier.get_name()), "Classifer")

            if self.cache:
                logging.info("Trained models ready in directory %s" % file_context.cache_dir)

            end = time.time()
            
            diff = end-start
            logging.info("Elapsed time: %0.2f s" % diff)
            logging.info("Average time: %0.2f s" % (diff/len(training_sets)))

Prediction = namedtuple("Prediction", ["preprocessor", "classifier", "predictions"])

class Predict(Task):
    def __init__(self, preprocessors, classifiers, generate_report=True, cache=True, overwrite=False):
        Task.__init__(self)
        self.preprocessors = preprocessors
        self.classifiers = classifiers
        self.generate_report = generate_report
        self.cache = cache
        self.overwrite = overwrite
    
    def requires(self):
        return [Train(self.preprocessors, self.classifiers, cache=self.cache, overwrite=self.overwrite), PreprocessTest(self.preprocessors, cache=self.cache, overwrite=self.overwrite)]
    
    def run(self, file_context, input_args):
        classifier_models, test_sets = input_args
        preprocessor_sets = defaultdict(list)
        for test_set in test_sets:
            preprocessor_sets[test_set.preprocessor.get_name()].append(test_set)
        classifier_sets = defaultdict(lambda: defaultdict(dict))
        for model in classifier_models:
            classifier_sets[model.preprocessor.get_name()][model.classifier.get_name()][model.target] = model.classifier
        
        #Match classifiers and test datasets with same target
        for preprocessor_name, test_set_partition in preprocessor_sets.iteritems():
            for classifier_name, models in classifier_sets[preprocessor_name].iteritems():
                # The submission format consists of clip name and preictal probability
                guesses = [('clip','preictal')]
                
                start = time.time()
                
                logging.info("Making predictions with preprocessor %s and classifier_name %s" % (preprocessor_name, classifier_name))
                
                for test_set in test_set_partition:
                    model = models[test_set.target]
                    prediction = model.predict(test_set.test)
                    assert len(test_set.name) == len(prediction)
                    guesses += [(test_set.name[i], prediction[i]) for i in range(0,len(prediction))]
                
                guesses[1:] = sorted(guesses[1:], key=lambda x: x[0])
                
                if self.generate_report:
                    filename = 'submission-%s_%s.csv' % (preprocessor_name, classifier_name)
                    filename = os.path.join(file_context.submission_dir, filename)
                    with open(filename, 'w') as f:
                        csv_writer = csv.writer(f)
                        csv_writer.writerows(guesses)
                    logging.info("Submission file %s ready in directory %s" % (os.path.basename(filename), os.path.dirname(filename)))
                else:
                    logging.info("Predictions: %s" % guesses)
                
                self.output.append(Prediction(preprocessor_name, classifier_name, guesses))
                
                end = time.time()
                
                diff = end-start
                logging.info("Elapsed time: %0.2f s" % diff)
                logging.info("Average time: %0.2f s" % (diff/len(test_sets)))

CrossValidation = namedtuple("CrossValidation", ["target", "preprocessor", "classifier", "cv"])

class CrossValidate(Task):
    def __init__(self, preprocessors, classifiers, splits=3, make_report=True, cache=True, overwrite=False):
        Task.__init__(self)
        self.preprocessors = preprocessors
        self.classifiers = classifiers
        self.splits = splits
        self.make_report = make_report
        self.cache = cache
        self.overwrite = overwrite
    
    def requires(self):
        return [PreprocessTraining(self.preprocessors, cache=self.cache, overwrite=self.overwrite)]
    
    def run(self, file_context, training_sets):
        preprocessor_sets = defaultdict(list)
        for training_set in training_sets:
            preprocessor_sets[training_set.preprocessor].append(training_set)
                    
        for classifier in self.classifiers:
            for preprocessor, training_set_partition in preprocessor_sets.iteritems():
                start = time.time()
                
                reports = []
                
                logging.info("Cross validating datasets with preprocessor %s and classifier %s and %d splits" % (preprocessor.get_name(), classifier.get_name(), self.splits))
                
                means = []
                stds = []
                
                for training_set in training_set_partition:
                    scores = classifier.cross_validate(training_set.interictal, training_set.preictal, self.splits)
                    self.output.append(CrossValidation(training_set.target, preprocessor, classifier, scores))
                    logging.info("Cross validation for target %s:\n%s" % (training_set.target, scores))
                    mean = scores.mean()
                    means.append(mean)
                    std = scores.std()
                    stds.append(std)
                    logging.info("Accuracy: %0.2f (+/- %0.2f)" % (mean, std * 2))
                    
                    if self.make_report:
                        reports.append([training_set.target] + list(scores))
                        
                logging.info("Mean accuracy: %0.2f (+/- %0.2f)" % (numpy.mean(means), numpy.mean(stds) * 2))
            
            if self.make_report:
                filename = 'crossvalidation-%s_%s.csv' % (preprocessor.get_name(), classifier.get_name())
                filename = os.path.join(file_context.report_dir, filename)
                with open(filename, 'w') as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerows(reports)
                logging.info("Cross validation report %s ready in directory %s" % (os.path.basename(filename), os.path.dirname(filename)))
            
            end = time.time()
            diff = end-start
            logging.info("Elapsed time: %0.2f s" % diff)
            logging.info("Average time: %0.2f s" % (diff/len(training_sets)))

class Statistics(Task):
    def __init__(self, preprocessors, make_report=True, cache=True, overwrite=False):
        Task.__init__(self)
        self.preprocessors = preprocessors
        self.make_report = make_report
        self.cache = cache
        self.overwrite = overwrite
    
    def requires(self):
        return [PreprocessTraining(self.preprocessors, cache=self.cache, overwrite=self.overwrite)]
    
    def run(self, file_context, training_sets):
        preprocessor_sets = defaultdict(list)
        for training_set in training_sets:
            preprocessor_sets[training_set.preprocessor].append(training_set)
                    
        for preprocessor, training_set_partition in preprocessor_sets.iteritems():
            start = time.time()
            
            reports = []
                
            logging.info("Computing statistics for datasets with preprocessor %s" % preprocessor.get_name())
                
            for training_set in training_set_partition:
                reports += [[training_set.target] + s for s in statistical_analyze(training_set.interictal, training_set.preictal)]
            
            if self.make_report:
                filename = 'statistics-%s.csv' % preprocessor.get_name()
                filename = os.path.join(file_context.report_dir, filename)
                with open(filename, 'w') as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerows(reports)
                logging.info("Statistics report %s ready in directory %s" % (os.path.basename(filename), os.path.dirname(filename)))
            
            end = time.time()
            diff = end-start
            logging.info("Elapsed time: %0.2f s" % diff)
            logging.info("Average time: %0.2f s" % (diff/len(training_sets)))
    
class FeatureImportance(Task):
    def __init__(self, preprocessors, make_report=True, overwrite=False):
        Task.__init__(self)
        self.preprocessors = preprocessors
        self.make_report = make_report
        self.overwrite = overwrite
    
    def requires(self):
        return [PreprocessTraining(self.preprocessors, overwrite=self.overwrite)]
    
    def run(self, file_context, training_sets):
        preprocessor_sets = defaultdict(list)
        for training_set in training_sets:
            preprocessor_sets[training_set.preprocessor].append(training_set)
                
        classifier = ExtraTrees()
            
        for preprocessor, training_set_partition in preprocessor_sets.iteritems():
            start = time.time()
            
            reports = []
                
            logging.info("Computing feature importance for datasets with preprocessor %s" % preprocessor.get_name())
                
            for training_set in training_set_partition:
                classifier.train(training_set.interictal, training_set.preictal)
                importance = classifier.feature_importance()
                logging.info("Feature importance for target %s: %s" % (training_set.target, str(importance)))
                reports.append([training_set.target] + list(importance))
            
            if self.make_report:
                filename = 'feature_importance-%s.csv' % preprocessor.get_name()
                filename = os.path.join(file_context.report_dir, filename)
                with open(filename, 'w') as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerows(reports)
                logging.info("Feature importance report %s ready in directory %s" % (os.path.basename(filename), os.path.dirname(filename)))
            
            end = time.time()
            diff = end-start
            logging.info("Elapsed time: %0.2f s" % diff)
            logging.info("Average time: %0.2f s" % (diff/len(training_sets)))
            
"""
class SelectKFeatures(Task):
    def __init__(self, features, k=10, make_report=True):
        Task.__init__(self)
        self.preprocessor = Preprocessor(features, annotated=True)
        self.selector = KBestSelector(k)
        self.make_report = make_report
    
    def run(self, file_context, __):
        for target in file_context.targets:
            name = os.path.basename(target)
            logging.info("Preprocessing test datasets for target %s with preprocessor %s" % (name, self.preprocessor.get_name()))
            interictal = file_context.get_interictal_segments(target)
            processed_interictal = preprocess(interictal, self.preprocessor, file_context)
            preictal = file_context.get_interictal_segments(target)
            processed_preictal = preprocess(preictal, self.preprocessor, file_context)
            logging.info("Selecting features for target %s with preprocessor %s" % (name, self.preprocessor.get_name()))
            self.selector.select(processed_preictal, processed_interictal)
"""

class Plot(Task):
    def __init__(self, preprocessors, axis1, axis2, make_report=True, cache=True, overwrite=False):
        Task.__init__(self)
        self.preprocessors = preprocessors
        self.make_report = make_report
        self.axis1 = axis1
        self.axis2 = axis2
        self.cache = cache
        self.overwrite = overwrite
        
    def requires(self):
        return [PreprocessTraining(self.preprocessors, cache=self.cache, overwrite=self.overwrite)]
        
    def run(self, file_context, training_sets):
        preprocessor_sets = defaultdict(list)
        for training_set in training_sets:
            preprocessor_sets[training_set.preprocessor].append(training_set)
            
        for preprocessor, training_set_partition in preprocessor_sets.iteritems():
            start = time.time()
            
            logging.info("Plotting datasets with preprocessor %s" % preprocessor.get_name())
                
            for training_set in training_set_partition:
                plot(training_set.interictal, training_set.preictal, self.axis1, self.axis2)
            
            end = time.time()
            diff = end-start
            logging.info("Elapsed time: %0.2f s" % diff)
            logging.info("Average time: %0.2f s" % (diff/len(training_sets)))