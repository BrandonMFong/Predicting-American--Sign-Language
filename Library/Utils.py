"""
Utils
===============
My utility library 
"""
__author__ = "Brando"

from atpbar     import atpbar
from sys        import platform 
from datetime   import datetime
import os 
import pickle 
import numpy as np 
import pandas as pd 
import math 
import tensorflow as tf

YES = True 
NO  = False
fsSeparator = "\\" if platform == "win32" else "/"

kAlphabet = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
]

def GetPercentage(val):
    result = val * 100
    result = round(result)
    return result

class Logger():
    """
    Must declare at each 
    """
    PINK = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

    def __init__(self,scriptName=None) -> None:
        if scriptName is None: 
            self._header = "{date} - {logType}" 
        else: 
            self._header = scriptName + ": {date} - {logType}"
        self._datetimeFormat = "%d/%m/%Y %H:%M:%S"

    def Fatal(self,*args,**kargs):
        self._header = self._header.format(date=datetime.now().strftime(self._datetimeFormat), logType="fatal:")
        print(self._header, *args, **kargs)

    def Error(self,*args,**kargs):
        self._header = self._header.format(date=datetime.now().strftime(self._datetimeFormat), logType="error:")
        print(self._header, *args, **kargs)

    def Warn(self,*args,**kargs):
        self._header = self._header.format(date=datetime.now().strftime(self._datetimeFormat), logType="warning:")
        print(self._header, *args, **kargs)

    def Write(self,end=None,*args,**kargs):
        self._header = "{scriptName}: {date}:"
        self._header = self._header.format(date=datetime.now().strftime(self._datetimeFormat))
        print(self._header,*args,**kargs,end=end)

    def Except(self,*args,**kargs):
        self._header = self._header.format(date=datetime.now().strftime(self._datetimeFormat), logType="exception:")
        print(self._header, *args, **kargs)

class Base():
    """
    Base class for everything there has to do with data sets in CES 514 
    """

    _reshapeFacialKeyImage = (96,96)

    def __init__(self,dataSet=None,task1DataSet=None,task2DataSet=None,dropNaN=True) -> None:
        
        self._task1DataSet = None if task1DataSet is None else task1DataSet 
        self._task2DataSet = None if task2DataSet is None else task2DataSet 

        # Data set var 
        if dataSet is None: 
            self._dataSet = pd.DataFrame()
        else:
            self._dataSet = dataSet

        if dropNaN:
            self._dataSet = self._dataSet.dropna()

        # Train and Target Columns
        self._trainColumns = None 
        self._targetColumns = None 
        self._log = Logger()

        # PCA Column variable
        self._pcaColumns = []

    def CreateTrainAndTargetColumns(self,targetColumns: list) -> bool:
        """
        Creates Train and Target column from targetColumns list variable 
        """
        result = True 
        if result:
            if self._dataSet.empty:
                result = False 
        if result:
            self._targetColumns = targetColumns
            self._trainColumns = self._dataSet.drop(self._targetColumns, axis=1).columns
            if len(self._trainColumns) == 0 or len(self._targetColumns) == 0:
                result = False  
        return result

    def CreatePCAColumns(self,numComponents):
        for n in range(numComponents):
            self._pcaColumns.append("C{}".format(n))

    def FitImageColumn(self,column: str,convertToTensor=False):
        """
        Creates the appropriate object type for the image column for the Facial key points column
        """
        okayToContinue = True 
        if okayToContinue:
            okayToContinue = column in self._dataSet.columns
            if okayToContinue is False:
                self._log.Error(self.FitImageColumn.__name__, ": column", column, "does not exist in dataframe")
        if okayToContinue:
            if convertToTensor:
                result = self._dataSet[column].apply(
                    lambda image: 
                        tf.convert_to_tensor(np.asarray(np.fromstring(image, sep=" ")).astype('float32'))
                )
            else:
                result = self._dataSet[column].apply(lambda image: np.asarray(np.fromstring(image, sep=" ")).astype('float32'))
        if okayToContinue is False:
            result = None

        return result 

    def GetNewShape(self,length: int):
        result = () 
        square = math.sqrt(length)
        if math.remainder(square,1) == 0.0:
            square = int(square)
            result = (square, square)
        else:
            self._log.Error("input size not acceptable:", length)
            result = None 
        return result

class FileSystem():

    def GetFilePath(relativePath):
        return os.path.abspath(relativePath)

    def GetFileList(directoryPath, useDoc2Vec=False):
        """
        Parameters
        ----------------
        - directoryPath: Directory path to files
        """
        status = True 
        result = None
        log = Logger()

        if useDoc2Vec:
            pass 
        else:
            result = list()
            if status: 
                status = os.path.exists(directoryPath)
                if status is False:
                    log.Warn("Directory", directoryPath, "does not exist")

            if status: 
                for file in atpbar(os.listdir(directoryPath), name="Reading files"):
                    fullFilePath = directoryPath + "/" + file 
                    result.append(open(fullFilePath).read())

        return result
        
class InitError(Exception):
    """
    Error raise for Init procedures 
    """

    _message = "Error in initialization"

    def __init__(self,objectClass):
        self._class = objectClass
        super().__init__(self._message)

    def __str__(self):
        return f'{self._class}: init: {self._message}'
        
class TaskError(Exception):
    """
    Task Error 
    """

    _message = "Error in Task"

    def __init__(self,objectClass,additionalInfo=None):
        self._class = objectClass
        self._additionalInfo = additionalInfo
        super().__init__(self._message)

    def __str__(self):
        if self._additionalInfo is None:
            self._additionalInfo = ""
        else:
            self._additionalInfo = ": " + self._additionalInfo
        return f'{self._class}: Task: {self._message} {self._additionalInfo}'

class FunctionLibrary():
    
    kFind = 0
    kReplace = 1

    def centeroidnp(arr):
        length = arr.shape[0]
        sum_x = np.sum(arr[:, 0])
        sum_y = np.sum(arr[:, 1])
        return sum_x/length, sum_y/length

    @staticmethod
    def Load(filename):
        """
        Loads save variable in cache file if exists
        Returns None if it does not exist 
        """
        result = None
        if os.path.exists(filename):
            result = pickle.load(open(filename, "rb"))[0]
        return result

    @staticmethod 
    def LoadWithCallback(filename, callback):
        result = None 
        if os.path.exists(filename):
            result = pickle.load(open(filename, "rb"))[0]
            
        callback(result)

    @staticmethod
    def Save(variable, filename):
        """
        Saves variable into cache file in the current directory 
        """
        pickle.dump([variable], open(filename, "wb"))

    @staticmethod 
    def StringToCharList(string):
        result = list()
        for char in string:
            result.append(char)
        return result 

    @staticmethod 
    def StripCharacters(string, charactersList: list):
        for char in charactersList:
            string = string.replace(char[FunctionLibrary.kFind], char[FunctionLibrary.kReplace])
        return string 