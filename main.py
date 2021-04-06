# Brando 
# 3/30/2021
# Final Project

from numpy.core.fromnumeric import reshape
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import sys
import os
import math
import time
from progress.bar import Bar
import threading
import time, random
from atpbar import atpbar

from tensorflow.python.util.nest import flatten_with_joined_string_paths

class xASLHandler():

    _basePath = sys.path[0]

    # Data
    _rawTestFile = "/data/test/sign.csv"
    _rawTrainFile = "/data/train/sign.csv"

    def __init__(self) -> None:
        okayToContinue = True 
        fullTestFilename = None
        fullTrainFilename = None
        self._trainData = None
        self._testData = None
        self._targetColumn = None
        self._trainColumns = None
        self._testImages = None
        self._trainImages = None
        self._imageTestArray = None
        self._imageTrainArray = None
        temp = None
        newShape = None
        check = False

        if okayToContinue:
            fullTestFilename = self._basePath + self._rawTestFile
            if os.path.exists(fullTestFilename) is False:
                okayToContinue = False
                print("File", fullTestFilename, "does not exist")
            fullTrainFilename = self._basePath + self._rawTrainFile
            if os.path.exists(fullTrainFilename) is False:
                okayToContinue = False
                print("File", fullTrainFilename, "does not exist")

        if okayToContinue:
            self._testData = pd.read_csv(fullTestFilename)
            self._trainData = pd.read_csv(fullTrainFilename)
            okayToContinue = True if self._testData.empty is False and self._trainData.empty is False else False 

        if okayToContinue:
            self._targetColumn = "label"
            self._trainColumns = self._trainData.drop([self._targetColumn], axis=1).columns
            okayToContinue = True if self._trainColumns is not None else False 

        # Organize into reshaped datasets
        if okayToContinue:
            temp = math.sqrt(self._trainData[self._trainColumns].shape[1])
            if math.remainder(temp,1) == 0.0:
                temp = int(temp)
                newShape = (temp, temp)
            else:
                print("Error in reading data")
                okayToContinue = False
            
        # TODO separate progress bars 
        if okayToContinue:
            # Test
            loadImageTestArrayThread = threading.Thread(target=self.loadImageTestArrayOnThread, args=(newShape,))
            loadImageTestArrayThread.start()

            # Train
            print()
            loadImageTrainArrayThread = threading.Thread(target=self.loadImageTrainArrayOnThread, args=(newShape,))
            loadImageTrainArrayThread.start()

            loadImageTestArrayThread.join()
            loadImageTrainArrayThread.join()


    def loadImageTestArrayOnThread(self,newShape):
        for i in atpbar(range(self._testData.shape[0]), name="Test Data"):
            row = self._testData.iloc[i]
            data = np.array(row[self._trainColumns])
            reshapedData = data.reshape(newShape)
            if self._imageTestArray is None:
                self._imageTestArray = np.array([reshapedData])
            else:
                self._imageTestArray = np.concatenate((self._imageTestArray, [reshapedData]))

    def loadImageTrainArrayOnThread(self,newShape):
        for i in atpbar(range(self._trainData.shape[0]), name="Train Data"):
            row = self._testData.iloc[i]
            data = np.array(row[self._trainColumns])
            reshapedData = data.reshape(newShape)
            if self._imageTrainArray is None:
                self._imageTrainArray = np.array([reshapedData])
            else:
                self._imageTrainArray = np.concatenate((self._imageTrainArray, [reshapedData]))

    def Run(self):
        print(self._imageTestArray)

class Project():
    """
    Final Project
    =============
    Author: Brando 

    Overview 
    -------------
    My project is inspired by https://www.youtube.com/watch?v=pDXdlXlaCco 
    Dataset: https://www.kaggle.com/datamunge/sign-language-mnist

    """

    _basePath = sys.path[0]

    def __init__(self) -> None:
        self._signLangHandler = xASLHandler()
    
    def Do(self):
        self._signLangHandler.Run()

if __name__ == "__main__":
    print(Project.__doc__)

    project = Project()

    project.Do()
    
