'''
Created on Aug 30, 2014

@author: Michael
'''
import logging
import os
import sys

from seizure.classifier import NeuralNetwork, ClassifierPipeline, SVM, RandomForest, DecisionTree, ExtraTrees, RFECVSelector, VarianceSelector, KBestSelector
from seizure.files import FileContext
from seizure.preprocessor import Preprocessor
from seizure.transform import PCASTFTBands, ICASTFTBands, Magnitude, TimeBandPower, TimeBandPowerDiff, UpperTriangle, Eigenvalues, PowerSpectrumCorrelation, FrequencyBandPower, Correlation, FrequencyCorrelation, STFTBands, TimeCorrelation, Log10
from seizure.task import CrossValidate, FeatureImportance, TaskManager, Predict, PreprocessTraining, Plot, Train, Statistics

# Creates a symlink to the directory above this one
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
# Location of settings file that specifies file locations
SETTINGS_FILE = "SETTINGS.json"

def get_context():
    return FileContext.from_json(os.path.join(ROOT_DIR, SETTINGS_FILE))

def run_tasks(tasks):
    file_context = get_context()
    #file_context.set_limit(20)
    
    logging.basicConfig(filename=file_context.log_file, level=logging.DEBUG)
    root = logging.getLogger()
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    root.addHandler(ch)
    
    manager = TaskManager(tasks, file_context)
    manager.run()

def main():
    preprocessors = [
      Preprocessor([UpperTriangle(FrequencyCorrelation()), UpperTriangle(TimeCorrelation()), Log10(FrequencyBandPower())], "p7"),
      #Preprocessor([TimeOutliers()], "outliers2")
      #Preprocessor([Log10(Magnitude(STFTBands()))], "power_test_5"),
      #Preprocessor([PCASTFTBands()], "pca_stft_bands"),
      #Preprocessor([Log10(Magnitude(ICASTFTBands()))], "ica_stft_bands"),
    ]
    
    classifiers = [
                   SVM(),
    #               SVM(kernel='linear'),
    #               RandomForest(),
    #               DecisionTree(),
    #ClassifierPipeline(SVM(), KBestSelector(k=10)),
    #ClassifierPipeline(SVM(), KBestSelector(k=5)),
    #ClassifierPipeline(SVM(kernel='linear'), KBestSelector(k=10)),
    #ClassifierPipeline(SVM(kernel='linear'), KBestSelector(k=5)),
    #ClassifierPipeline(RandomForest(), KBestSelector(k=10)),
    #ClassifierPipeline(RandomForest(), KBestSelector(k=5)),
    #ClassifierPipeline(DecisionTree(), KBestSelector(k=10)),
    #ClassifierPipeline(DecisionTree(), KBestSelector(k=5)),
    #ClassifierPipeline(ExtraTrees(), KBestSelector(k=10)),
    #ClassifierPipeline(ExtraTrees(), KBestSelector(k=5)),
    #ClassifierPipeline(SVM(), RFECVSelector()),
    #ClassifierPipeline(RandomForest(), RFECVSelector()),
    #ClassifierPipeline(DecisionTree(), RFECVSelector()),
    #ClassifierPipeline(ExtraTrees(), RFECVSelector()),
    #ClassifierPipeline(SVM(kernel="sigmoid"), NeuralNetwork(n_components=2400)),
    #ClassifierPipeline(SVM(kernel="sigmoid"), NeuralNetwork(n_components=2400,learning_rate=0.5)),
    #ClassifierPipeline(SVM(kernel="sigmoid"), NeuralNetwork(n_components=2400,learning_rate=0.05)),
    #ClassifierPipeline(SVM(kernel="linear"), NeuralNetwork(n_components=2400)),
    #ClassifierPipeline(SVM(kernel="linear"), NeuralNetwork(n_components=2400,learning_rate=0.5)),
    #ClassifierPipeline(SVM(kernel="linear"), NeuralNetwork(n_components=2400,learning_rate=0.05)),
    #ClassifierPipeline(RandomForest(), NeuralNetwork()),
    #ClassifierPipeline(RandomForest(), NeuralNetwork(learning_rate=0.5)),
    #ClassifierPipeline(RandomForest(), NeuralNetwork(learning_rate=0.05)),
    ]
    
    tasks = [CrossValidate(preprocessors, classifiers)]
    #tasks = [Statistics(preprocessors), CrossValidate(preprocessors, classifiers), Predict(preprocessors, classifiers)]
    #tasks = [Statistics(preprocessors, cache=False)]
    #tasks = [CrossValidate(preprocessors, classifiers, make_report=False)]
    
    run_tasks(tasks)

if __name__ == "__main__":
    main()