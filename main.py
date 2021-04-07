# Brando 
# 3/30/2021
# Final Project

from tensorflow import keras
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import sys
import os
import math
import threading
from atpbar import atpbar
import pickle
from os import path


def load(filename):
    result = None
    found = False 
    # Get Data
    if path.exists(filename):
        result = pickle.load(open(filename, "rb"))[0]
        found = True 

    return result, found

def save(variable, filename):
    pickle.dump([variable], open(filename, "wb"))

class xASLHandler():

    _basePath = sys.path[0]

    # Data
    _rawTestFile = "/data/test/sign.csv"
    _rawTrainFile = "/data/train/sign.csv"

    # Cache files
    _testCache = "testImages.cache"
    _trainCache = "trainImages.cache"

    def __init__(self) -> None:
        okayToContinue = True 
        fullTestFilename = None
        fullTrainFilename = None
        temp = None
        newShape = None
        testCacheFound = False
        trainCacheFound = False

        self._trainData = None
        self._testData = None
        self._targetColumn = None
        self._trainColumns = None
        self._testImages = None
        self._trainImages = None
        self._imageTestArray = None
        self._imageTrainArray = None

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
            
        if okayToContinue:
            self._imageTestArray, testCacheFound    = load(self._testCache)
            self._imageTrainArray, trainCacheFound  = load(self._trainCache)

            if testCacheFound is not True:
                # Test
                print("Image Test cache was not found, creating data...")
                loadImageTestArrayThread = threading.Thread(target=self.loadImageTestArrayOnThread, args=(newShape,))
                loadImageTestArrayThread.start()

            # Train
            if trainCacheFound is not True:
                print("Image Train cache was not found, creating data...")
                loadImageTrainArrayThread = threading.Thread(target=self.loadImageTrainArrayOnThread, args=(newShape,))
                loadImageTrainArrayThread.start()

            if trainCacheFound is not True:
                loadImageTrainArrayThread.join()

            if testCacheFound is not True:
                loadImageTestArrayThread.join()

            okayToContinue = True if self._imageTestArray is not None and self._imageTrainArray is not None else False 
        
        if ~okayToContinue:
            raise RuntimeError("Error: problem during initialization")

    def loadImageTestArrayOnThread(self,newShape):
        for i in atpbar(range(self._testData.shape[0]), name="Test Data"):
            row = self._testData.iloc[i]
            data = np.array(row[self._trainColumns])
            reshapedData = data.reshape(newShape)
            if self._imageTestArray is None:
                self._imageTestArray = np.array([reshapedData])
            else:
                self._imageTestArray = np.concatenate((self._imageTestArray, [reshapedData]))
        
        save(self._imageTestArray, self._testCache)

    def loadImageTrainArrayOnThread(self,newShape):
        for i in atpbar(range(self._testData.shape[0]), name="Train Data"):
            row = self._trainData.iloc[i]
            data = np.array(row[self._trainColumns])
            reshapedData = data.reshape(newShape)
            if self._imageTrainArray is None:
                self._imageTrainArray = np.array([reshapedData])
            else:
                self._imageTrainArray = np.concatenate((self._imageTrainArray, [reshapedData]))
        
        save(self._imageTrainArray, self._trainCache)

    def Run(self):
        print(self._imageTestArray)
        plt.imshow(self._imageTestArray[0])
        plt.show()

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
    
