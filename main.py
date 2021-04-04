# Brando 
# 3/30/2021
# Final Project

from numpy.core.fromnumeric import reshape
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import sys
import os

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
            temp =  self._trainData[self._trainColumns].shape[1] ** 2
            if type(temp) == int:
                newShape = (temp, temp)
            else:
                print("Error in reading data")
                okayToContinue = False
            
            # Test
            self._imageTestArray = np.array([temp,temp,1])
            for row in self._testData.iterrows():
                key = row[self._targetColumn]
                data = np.array(row[self._trainColumns])
                reshapedData = data.reshape(newShape)

                if check is False:
                    self._imageTestArray = np.concatenate([self._imageTestArray], reshapedData)
                    check = False
                else:
                    self._imageTestArray = np.concatenate(self._imageTestArray, reshapedData)

    def Run(self):
        arr = np.array(self._testData.loc[0, self._trainColumns])
        print(arr.reshape((28,28)))
        plt.imshow(arr.reshape((28,28)))
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
    
