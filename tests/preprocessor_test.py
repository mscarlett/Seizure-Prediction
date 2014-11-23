'''
Created on Sep 9, 2014

@author: Michael
'''
import sys
import unittest

from seizure.main import get_context, run_tasks
from seizure.preprocessor import Preprocessor
from seizure.task import PreprocessTraining
from seizure.transform import TimeOutliers, STFT, UpperTriangle, FrequencyCorrelation, TimeCorrelation

class PreprocessTest(unittest.TestCase):
    
    def setUp(self):
        preprocessors = [
              Preprocessor([STFT()], "p5"),
            ]

        file_context = get_context()
        file_context.set_limit(20)
        
        tasks = [PreprocessTraining(preprocessors, cache=False)]
        
        run_tasks(tasks)
        
    def testName(self):
        self.assertTrue(True)
        
    def tearDown(self):
        pass
    
if __name__ == "__main__":
    sys.argv = ['PreprocessTest.testName']
    unittest.main()