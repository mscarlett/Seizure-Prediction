'''
Created on Sep 6, 2014

@author: Michael
'''
from collections import namedtuple, OrderedDict
import itertools
import logging
import multiprocessing
import numpy
import os
import scipy
import time

from seizure.graph import DependencyGraph

def preprocess(datasets, preprocessor, file_context, nsplits=10, cache=True, overwrite=False):
    logging.info("Preprocessing %d datasets" % len(datasets))
    results = []
    start = time.time()
    
    #preprocess_mp(datasets, preprocessor, file_context, nsplits, cache, overwrite)
    
    for sample in datasets:
        #Check if cached result exists already
        cache_name = "%s-%s" % (preprocessor.get_name(), os.path.basename(sample))
        feature_set = file_context.cache_load(cache_name, "preprocessed") if cache and not overwrite else None
        if feature_set is None:
            feature_set = []
            for dataset in split_dataset(load_dataset(load_mat(sample)), nsplits):
                feature_set.append(preprocessor.apply(dataset))
            if cache:
                file_context.cache_dump(feature_set, cache_name, "preprocessed")
        results += feature_set
    
        
    end = time.time()
    diff = end - start
    logging.info("Finished preprocessing %d datasets" % len(datasets))
    logging.info("Elapsed time: %0.2f s" % diff)
    logging.info("Average time: %0.2f s" % (diff/len(datasets)))
    return results
"""
def preprocess_mp(datasets, preprocessor, file_context, nsplits, cache, overwrite):
    def get_preprocessor():
        return preprocessor
    
    def get_file_context():
        return file_context
    
    p = multiprocessing.Pool(multiprocessing.cpu_count())
    results = p.map(preprocess_worker, [(dataset, get_preprocessor, get_file_context, nsplits, cache, overwrite) for dataset in datasets])
    output = []
    for result in results:
        output.append(result())
    return list(itertools.chain(*output))

def preprocess_worker(sample, get_preprocessor, get_file_context, nsplits, cache, overwrite):
    preprocessor = get_preprocessor()
    file_context = get_file_context()
    #Check if cached result exists already
    cache_name = "%s-%s" % (preprocessor.get_name(), os.path.basename(sample))
    feature_set = file_context.cache_load(cache_name, "preprocessed") if cache and not overwrite else None
    if feature_set is None:
        feature_set = []
        for dataset in split_dataset(load_dataset(load_mat(sample)), nsplits):
            feature_set.append(preprocessor.apply(dataset))
        if cache:
            file_context.cache_dump(feature_set, cache_name, "preprocessed")
            
    def get_results():
        return feature_set
            
    return get_results
"""

def load_mat(filename):
    with open(filename, "r") as f:
        return scipy.io.loadmat(f)

"""
Represents an annotated raw data set.
"""
DataSet = namedtuple("DataSet", ["data", "data_length_sec", "sampling_rate", "channels", "sequence", "name"])
    
def load_dataset(input_data):
    for key in input_data.keys():
        if key not in ('__version__', '__header__', '__globals__'):
            raw_data = input_data[key][0][0]
            data = raw_data[0]
            data_length_sec = raw_data[1]
            sampling_rate = raw_data[2]
            channel_names = raw_data[3] if len(raw_data) > 3 else None
            sequence = raw_data[4] if len(raw_data) > 4 else None
            return DataSet(data, data_length_sec, sampling_rate, channel_names, sequence, key)
    raise ValueError("Cannot parse MATLAB data")

def split_dataset(dataset, nsplits):
    if nsplits <= 1:
        yield dataset
        return
    split_data = numpy.array_split(dataset.data, nsplits,axis=1)
    new_time = dataset.data_length_sec/nsplits
    for data in split_data:
        yield DataSet(data, new_time, dataset.sampling_rate, dataset.channels, dataset.sequence, dataset.name)

class Preprocessor(DependencyGraph):
    """
    Generate the feature set from the input data by constructing a dependency graph of transformations
    for obtaining output data. The dependency graph is defined as the digraph with unique vertices representing transforms
    and edges that represent input-output pairs of transforms.
    """
    def __init__(self, transforms, name):
        DependencyGraph.__init__(self, transforms, name)
        
    def get_output(self, results):
        """
        Transforms the input results into the feature set.
        """
        outputs = FeatureSet()
        for transform in self.final_transforms:
            v = self.get(transform)
            outputs.append(results[v], v["transform"].get_name())
        del results #Erases graph results once we don't need it
        return outputs
        
class FeatureSet(numpy.ndarray):
    """
    Extend numpy.ndarray with methods for retrieving individual features
    """
    
    def __new__(cls):
        instance = numpy.ndarray.__new__(cls, **{'shape': (0,)})
        setattr(instance, 'features', OrderedDict())
        assert hasattr(instance, 'features')
        return instance
    
    def append(self, feature, name):
        """
        Adds a new feature. This should only be called when there are not multiple
        references to this object.
        """
        feature = numpy.ndarray.flatten(feature)
        self.resize(len(self) + len(feature), refcheck=False)
        self[-len(feature):] = feature
        self.features[name] = len(feature)
        
    def get_features(self):
        """
        Create a (name, feature) dictionary from this array
        """
        features = OrderedDict()
        index = 0
        for name, length in self.features.iteritems():
            features[name] = self[index:index+length]
            index += length
        return features
    
    def get_feature_indices(self):
        indices = OrderedDict()
        index = 0
        for name, length in self.features.iteritems():
            indices[name] = (index,index+length)
            index += length
        return indices
            
    def iterfeatures(self):
        """
        Iterate over (name, feature) pairs in the array
        """
        index = 0
        for name, length in self.features.iteriems():
            yield name, self[index:index+length]
            index += length
