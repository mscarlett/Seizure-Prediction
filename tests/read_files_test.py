'''
Created on Sep 4, 2014

@author: newuser
'''
import sys
import unittest

from seizure.main import get_context

class ReadFilesTest(unittest.TestCase):


    def setUp(self):
        self.context = get_context()


    def tearDown(self):
        pass


    def testName(self):
        self.assertTrue(True)

if __name__ == "__main__":
    sys.argv = ['', 'ReadFilesTest.testName']
    unittest.main()