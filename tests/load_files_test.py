'''
Created on Sep 4, 2014

@author: newuser
'''
import sys
import unittest

from seizure.main import get_context

class LoadFilesTest(unittest.TestCase):

    def setUp(self):
        self.context = get_context()

    def tearDown(self):
        pass


    def testDirectoryExists(self):
        self.assertTrue(True)
        
    def testInterictalFiles(self):
        interictal = []
        for target in self.context.targets:
            interictal += self.context.get_interictal_segments(target)
        self.assertTrue(interictal != [])
        
    def testPreictalFiles(self):
        preictal = []
        for target in self.context.targets:
            preictal = self.context.get_preictal_segments(target)
        self.assertTrue(preictal != [])

if __name__ == "__main__":
    sys.argv = ['', 'LoadFilesTest.testDirectoryExists', 'LoadFilesTest.testInterictalFiles', 'LoadFilesTest.testPreictalFiles']
    unittest.main()