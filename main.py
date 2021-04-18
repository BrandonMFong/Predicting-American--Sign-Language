# Brando 
# 3/30/2021
# Final Project

from    tensorflow          import keras
from    atpbar              import atpbar
from    os                  import path
from    Library             import FunctionLibrary  as fLib
from    Library             import FileSystem       as fs 
from    Library             import Logger           as log
from    Library             import InitError
from    Library             import Base
import  pandas              as pd 
import  numpy               as np
import  matplotlib.pyplot   as plt 
import  sys
import  os
import  math
import  threading
import  pickle

class xASLHandler():

    # Data
    _rawTestFile = "/data/test/sign.csv"
    _rawTrainFile = "/data/train/sign.csv"

    # Cache files
    _testCache = "testImages.cache"
    _trainCache = "trainImages.cache"

    # Label Mapping 
    _labelDictionary = {
        0 : "A",
        1 : "B", 
        2 : "C", 
        3 : "D", 
        4 : "E", 
        5 : "F", 
        6 : "G", 
        7 : "H", 
        8 : "I", 
        9 : "J", 
        10 : "K", 
        11 : "L", 
        12 : "M", 
        13 : "N", 
        14 : "O", 
        15 : "P",
        16 : "Q", 
        17 : "R", 
        18 : "S", 
        19 : "T", 
        20 : "U", 
        21 : "V", 
        22 : "W", 
        23 : "X", 
        24 : "Y", 
        25 : "Z"
    }

    # Specifically for this dataset for ASL 
    _defaultTargetColumn = "label"

    _defaultPaddingForModel = "same"

    def __init__(self) -> None:
        okayToContinue      = True 
        fullTestFilename    = None
        fullTrainFilename   = None
        temp                = None
        newShape            = None
        inputShapeModel     = None 
        outputShapeModel    = None

        self._trainData         = None
        self._testData          = None
        self._testImages        = None
        self._trainImages       = None
        self._imageTestArray    = None
        self._imageTrainArray   = None
        self._model             = None 

        if okayToContinue:
            fullTestFilename = fs.CreateFilePath(self._rawTestFile)
            if os.path.exists(fullTestFilename) is False:
                okayToContinue = False
                print("File", fullTestFilename, "does not exist")
            fullTrainFilename = fs.CreateFilePath(self._rawTrainFile)
            if os.path.exists(fullTrainFilename) is False:
                okayToContinue = False
                print("File", fullTrainFilename, "does not exist")

        if okayToContinue:
            self._testData = Base(pd.read_csv(fullTestFilename))
            self._trainData = Base(pd.read_csv(fullTrainFilename))
            okayToContinue  = True if self._testData._dataSet.empty is False and self._trainData._dataSet.empty is False else False 

        if okayToContinue:
            okayToContinue = self._testData.CreateTrainAndTargetColumns(targetColumns=[self._defaultTargetColumn])

        if okayToContinue:
            okayToContinue = self._trainData.CreateTrainAndTargetColumns(targetColumns=[self._defaultTargetColumn])

        # Organize into reshaped datasets
        if okayToContinue:
            temp = math.sqrt(self._trainData._dataSet[self._trainData._trainColumns].shape[1])
            if math.remainder(temp,1) == 0.0:
                temp        = int(temp)
                newShape    = (temp, temp)
            else:
                print("Error in reading data")
                okayToContinue = False
            
        # Get the image arrays for test and train data 
        if okayToContinue:
            self._imageTestArray    = fLib.Load(self._testCache)
            self._imageTrainArray   = fLib.Load(self._trainCache)


            if self._imageTestArray is None:
                # Test
                print("Image Test cache was not found, creating data...")
                loadImageTestArrayThread = threading.Thread(target=self.LoadImageTestArrayOnThread, args=(newShape,))
                loadImageTestArrayThread.start()

            # Train
            if self._imageTrainArray is None:
                print("Image Train cache was not found, creating data...")
                loadImageTrainArrayThread = threading.Thread(target=self.LoadImageTrainArrayOnThread, args=(newShape,))
                loadImageTrainArrayThread.start()

            if self._imageTrainArray is None:
                loadImageTrainArrayThread.join()

            if self._imageTestArray is None:
                loadImageTestArrayThread.join()

            okayToContinue = True if self._imageTestArray is not None and self._imageTrainArray is not None else False 

        # Prepare variables for the Sequential model 
        if okayToContinue: 
            temp = math.sqrt(len(self._trainData._trainColumns))
            if math.remainder(temp,1) == 0.0:
                temp            = int(temp)
                # inputShapeModel = (temp, temp, 1)
                inputShapeModel = (temp, temp)
            else:
                log.Error("Incompatable size for input images.  Total pixels for dataset:", len(self._trainData._trainColumns))
                okayToContinue = False

        # Initialize the model 
        if okayToContinue:
            self._model = keras.Sequential()
            # inputShapeModel = (28,28,1)
            try:
                outputShapeModel = len(self._labelDictionary) - 2 # Not include J or Z
                self._model.add(keras.layers.Conv2D(32,(3,3),padding=self._defaultPaddingForModel,input_shape=inputShapeModel,activation=keras.activations.relu))
                self._model.add(keras.layers.MaxPool2D((2,2)))

                self._model.add(keras.layers.Conv2D(64,(3,3),padding=self._defaultPaddingForModel,activation=keras.activations.relu))
                self._model.add(keras.layers.MaxPool2D((2,2)))

                self._model.add(keras.layers.Conv2D(128,(3,3),padding=self._defaultPaddingForModel,activation=keras.activations.relu))
                self._model.add(keras.layers.MaxPool2D((2,2)))

                self._model.add(keras.layers.Flatten())
                self._model.add(keras.layers.Dense(512,activation=keras.activations.relu))
                self._model.add(keras.layers.Dense(outputShapeModel,activation=keras.activations.softmax))
            except TypeError as e:
                log.Fatal("Could not build keras model")
                log.Fatal("Exception message:\n", e)
                okayToContinue = False
            except ValueError as e:
                log.Fatal("Could not build keras model")
                log.Fatal("Exception message:", e)
                okayToContinue = False
            
        if okayToContinue is False:
            raise InitError(type(self))

    def LoadImageTestArrayOnThread(self,newShape):
        for i in atpbar(range(self._testData.shape[0]), name="Test Data"):
            row = self._testData.iloc[i]
            data = np.array(row[self._trainColumns])
            reshapedData = data.reshape(newShape)
            if self._imageTestArray is None:
                self._imageTestArray = np.array([reshapedData])
            else:
                self._imageTestArray = np.concatenate((self._imageTestArray, [reshapedData]))
        
        fLib.Save(self._imageTestArray, self._testCache)

    def LoadImageTrainArrayOnThread(self,newShape):
        for i in atpbar(range(self._trainData.shape[0]), name="Train Data"):
            row = self._trainData.iloc[i]
            data = np.array(row[self._trainColumns])
            reshapedData = data.reshape(newShape)
            if self._imageTrainArray is None:
                self._imageTrainArray = np.array([reshapedData])
            else:
                self._imageTrainArray = np.concatenate((self._imageTrainArray, [reshapedData]))
        
        fLib.Save(self._imageTrainArray, self._trainCache)

    def Run(self):
        """
        TODO create model 
        
        To Plot
        ------
        plt.imshow(self._imageTestArray[i])
        """
        index = 4
        print(self._labelDictionary[int(self._testData._dataSet[self._testData._targetColumns].loc[index])])
        plt.imshow(self._imageTestArray[index])
        plt.show()
        

class Project():
    """
    Final Project
    =============
    Author: Brando 

    Oultine 
    -------------
    The final image recognition project is following the structure of Chapter 2, that is the main steps to consider are:

    1. Look at the big picture.
    2. Get the data.
    3. Discover and visualize the data to gain insights.
    4. Prepare the data for Machine Learning algorithms.
    5. Select a model and train it.
    6. Fine-tune your model.

    Depending on the chosen project, It is not expected to complete the task that you set out to do but it is expected that at least one of the above steps challenges you, i.e. is different from what we have done in class. For example, getting the data could be challenging because it is not straightforward to scrape the images from the web, building a model could be changing the loss function of a neural network that is specific to the project.

    The finals week of class will be presenting the results of your project in a similar form to an article. In particular,

    - Intoduction. What is the problem you are trying to solve?

    - Dataset. present the details of dataset, displaying example images and how you obtained the dataset

    - Fine-Tuning. The different model parameters that were attempted

    - Results including accuracy tables and display the final outputs, i.e. how the trained network performed on some sample images

    - Outlook, discuss points of improvement or future steps to take the project further.

    I will be asking questions during the presentation to confirm your understanding in each step in the process.

    Details
    -------------
    Breaking down the outline:

    1. Look at the big picture.

    It is more motivating to choose a project that you personally find practical or that fills an obvious need.

    2. Get the data.

    Google Images has already done some hard work in labelling items but it is a good starting point if you need to get classified images. Note that scraping can be a challenge in itself, there is a bit of a hack using the package pyautogui, where you can give control of the keyword and mouse/cursor to the computer and perform the manual actions that you could take to save images from image search.

    There will be a challenge of creating a big enough dataset that is relatively unbiased. As you have observed, more examples lead to greater accuracy but gathering data can also be a big time suck so once you feel there is enough to continue with the following steps, then come back and gather even more data when you have prepared the code for the rest of the project.

    3. Discover and visualize the data to gain insights

    This stage of Exploratory Data Analysis (EDA) could be just a matter of viewing the images and the targets. If the images were scraped from Google Images, there may be some false positives and it would be good to crop them out. In the FacialKeypoints, the missing data there shared how unclean the data was but if you start to look at specific images, e.g. the shortest distance between eyes, you will find that there are many faces that didn't fill the whole 96x96 grid and may have skewed the learning/accuracy. Another reason to perform this step and be comfortable that the quality of the data going into the model is good.

    4& 5. Select a model, Prepare the data for Machine Learning algorithms. Train the model

    Steps 4 and 5 in the outline are not separable for me, i.e. you cannot prepare the data if you don't know the model that you are trying to fit or the loss you are trying to optimize.

    For example, the Yolo network has a very particular output/target and you would need to prepare the data when you have understood the neural network output.

    6. Fine-tune your model.

    Applying hyperparameter optimization to improve your accuracy. This shouldn't be too challenging because you should be reimplementing the code that was prepared for Week 7 Assignment.

    Example projects:

    - Street2Shop, matching an image of clothing off the street with a similar item in an online store

    - Grape Detection, identifying the location and number of grapes from an image

    - Make/Model Classification, classifying the make and model of a car from whatever angle

    - OCR (Optical Character Recognition), being able to convert the image of text into the string

    - Sky segmentation, being able to identify the sky in an image.

    - Helmet Detection, identifying whether motorcyclists or cyclists are wearing helmets.
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
    
