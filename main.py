# Brando 
# 3/30/2021
# Final Project

from    tensorflow                              import keras
from    atpbar                                  import atpbar
from    Library                                 import FunctionLibrary  as fLib
from    Library                                 import FileSystem       as fs 
from    Library                                 import Logger 
from    Library                                 import InitError
from    Library                                 import Base
from    tensorflow.keras.preprocessing.image    import ImageDataGenerator
from    sklearn.model_selection                 import train_test_split
import  pandas              as pd 
import  numpy               as np
import  matplotlib.pyplot   as plt 
import  tkinter             as tk
import  sys
import  os
import  math
import  threading
import  cv2
import  time 

log = Logger(scriptName=__file__)

class xASLHandler():
    """
    https://www.kaggle.com/sandeepbhogaraju/sign-language-identifier-100-accuracy-tf-model
    """

    # Data
    _rawTestFile = "data/test/sign.csv"
    _rawTrainFile = "data/train/sign.csv"

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

    _testArrayName = "test"
    _trainArrayName = "train"

    def __init__(self,epochs=10) -> None:
        okayToContinue      = True 
        fullTestFilename    = None
        fullTrainFilename   = None
        temp                = None
        # newShape            = None
        inputShapeModel     = None 
        outputShapeModel    = None

        self._trainData         = None
        self._testData          = None
        self._testImages        = None
        self._trainImages       = None
        self._imageTestArray    = None
        self._imageTrainArray   = None
        self._model             = None 
        self._reshapeValue      = None
        self._epochs            = epochs
        self._trainGenerator    = None
        self._testGenerator     = None
        self._xTrain            = None
        self._yTrain            = None 
        self._xTest             = None 
        self._yTest             = None 

        if okayToContinue:
            fullTestFilename = fs.GetFilePath(self._rawTestFile)
            if os.path.exists(fullTestFilename) is False:
                okayToContinue = False
                print("File", fullTestFilename, "does not exist")
            fullTrainFilename = fs.GetFilePath(self._rawTrainFile)
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
                temp                = int(temp)
                self._reshapeValue  = (temp, temp)
            else:
                print("Error in reading data")
                okayToContinue = False
            
        # Get the image arrays for test and train data 
        if okayToContinue:
            self._imageTestArray    = fLib.Load(self._testCache)
            self._imageTrainArray   = fLib.Load(self._trainCache)

            # Test data
            if self._imageTestArray is None:
                print("Image Test cache was not found, creating data...")
                loadImageTestArrayThread = threading.Thread(target=self.LoadImageArrayOnThread, args=(self._testArrayName,self._testData._dataSet))
                loadImageTestArrayThread.start()

            # Train data 
            if self._imageTrainArray is None:
                print("Image Train cache was not found, creating data...")
                loadImageTrainArrayThread = threading.Thread(target=self.LoadImageArrayOnThread, args=(self._trainArrayName,self._trainData._dataSet))
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
                temp                = int(temp)
                self._reshapeValue  = (temp, temp, 1)
                # inputShapeModel = (temp, temp)
            else:
                log.Error("Incompatable size for input images.  Total pixels for dataset:", len(self._trainData._trainColumns))
                okayToContinue = False

        # Initialize the model 
        if okayToContinue:
            self._model = keras.Sequential()
            try:
                outputShapeModel = len(self._labelDictionary) - 2 # Not include J or Z

                # self._model.add(keras.layers.Conv2D(16, (3,3), padding='same', activation=keras.activations.relu,input_shape=(28, 28, 1)))
                self._model.add(keras.layers.Conv2D(16, (3,3), padding='same', activation=keras.activations.relu,input_shape=self._reshapeValue))
                self._model.add(keras.layers.MaxPooling2D((2,2)))
                self._model.add(keras.layers.Conv2D(32, (3,3), padding='same', activation=keras.activations.relu))
                self._model.add(keras.layers.MaxPooling2D((2,2)))
                self._model.add(keras.layers.Conv2D(64, (3,3), padding='same', activation=keras.activations.relu))
                self._model.add(keras.layers.MaxPooling2D((2,2)))
                self._model.add(keras.layers.Conv2D(128, (3,3), padding='same', activation=keras.activations.relu))
                self._model.add(keras.layers.MaxPooling2D((2,2)))
                self._model.add(keras.layers.Flatten())
                self._model.add(keras.layers.Dense(64, activation=keras.activations.relu))
                self._model.add(keras.layers.Dense(outputShapeModel, activation=keras.activations.softmax))
            except TypeError as e:
                log.Fatal("Could not build keras model")
                log.Except(e)
                okayToContinue = False
            except ValueError as e:
                log.Fatal("Could not build keras model")
                log.Except(e)
                okayToContinue = False
            except Exception as e:
                log.Fatal("Unknown exception")
                log.Except(e)
                okayToContinue = False
        
        # Compile the model 
        if okayToContinue:
            try:
                self._model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics = ['accuracy'])
                self._model.summary()
            except ValueError as e:
                log.Fatal("Could not compile keras model")
                log.Except(e)
                okayToContinue = False
            except Exception as e:
                log.Fatal("Unknown exception")
                log.Except(e)
                okayToContinue = False

        # Train the model 
        # if okayToContinue:
        #     self._xTrain    = self._imageTrainArray.astype(float)
        #     self._yTrain    = self._trainData._dataSet[self._trainData._targetColumns].astype(float)
        #     self._xTest     = self._imageTestArray.astype(float)
        #     self._yTest     = self._testData._dataSet[self._testData._targetColumns].astype(float)

        #     self._xTrain, X_validate, self._yTrain, Y_validate = train_test_split(self._xTrain, self._yTrain, test_size = 0.2, random_state = 12345)
            
        #     train_datagen = ImageDataGenerator(
        #         rescale=1/255,rotation_range=45, width_shift_range=0.25,
        #         height_shift_range=0.15,shear_range=0.15, zoom_range=0.2, 
        #         fill_mode='nearest'
        #     )
        #     test_datagen            = ImageDataGenerator(rescale=1/255)
        #     valid_datagen           = ImageDataGenerator(rescale=1/255)

        #     self._trainGenerator    = train_datagen.flow(self._xTrain, self._yTrain, batch_size=32)
        #     self._testGenerator     = test_datagen.flow(self._xTest,self._yTest,batch_size=32)
        #     valid_generator         = valid_datagen.flow(X_validate,Y_validate,batch_size=32)
            
        #     self._model.fit(
        #         self._trainGenerator,
        #         epochs=self._epochs,
        #         validation_data=valid_generator,
        #         callbacks = [
        #             keras.callbacks.EarlyStopping(monitor='loss', patience=10),
        #             keras.callbacks.ModelCheckpoint(
        #                 filepath='/kaggle/working/',
        #                 monitor='val_accuracy',
        #                 save_best_only=True
        #             )
        #         ]
        #     )

        if okayToContinue is False:
            raise InitError(type(self))

    def LoadImageArrayOnThread(self,forArray,dataSet):
        images = []
        for i in atpbar(range(dataSet.shape[0]), name="{} Data".format(forArray)):
            row     = dataSet.iloc[i]
            image   = np.array_split(row[1:],28)
            images.append(image)

        if forArray == "train":
            self._imageTrainArray = np.array(images)
            self._imageTrainArray = np.expand_dims(self._imageTrainArray,axis=3)
            fLib.Save(self._imageTrainArray, self._trainCache)
        elif forArray == "test":
            self._imageTestArray = np.array(images)
            self._imageTestArray = np.expand_dims(self._imageTestArray,axis=3)
            fLib.Save(self._imageTestArray, self._testCache)

    def Train(self):
        self._xTrain    = self._imageTrainArray.astype(float)
        self._yTrain    = self._trainData._dataSet[self._trainData._targetColumns].astype(float)
        self._xTest     = self._imageTestArray.astype(float)
        self._yTest     = self._testData._dataSet[self._testData._targetColumns].astype(float)

        self._xTrain, X_validate, self._yTrain, Y_validate = train_test_split(self._xTrain, self._yTrain, test_size = 0.2, random_state = 12345)
        
        train_datagen = ImageDataGenerator(
            rescale=1/255,rotation_range=45, width_shift_range=0.25,
            height_shift_range=0.15,shear_range=0.15, zoom_range=0.2, 
            fill_mode='nearest'
        )
        test_datagen            = ImageDataGenerator(rescale=1/255)
        valid_datagen           = ImageDataGenerator(rescale=1/255)

        self._trainGenerator    = train_datagen.flow(self._xTrain, self._yTrain, batch_size=32)
        self._testGenerator     = test_datagen.flow(self._xTest,self._yTest,batch_size=32)
        valid_generator         = valid_datagen.flow(X_validate,Y_validate,batch_size=32)
        
        self._model.fit(
            self._trainGenerator,
            epochs=self._epochs,
            validation_data=valid_generator,
            callbacks = [
                keras.callbacks.EarlyStopping(monitor='loss', patience=10),
                keras.callbacks.ModelCheckpoint(
                    filepath='/kaggle/working/',
                    monitor='val_accuracy',
                    save_best_only=True
                )
            ]
        )

    def Run(self):
        """
        To Plot
        ------
        plt.imshow(self._imageTestArray[i])
        """
        # define a video capture object
        vid = cv2.VideoCapture(0)

        runWindowThread = threading.Thread(target=self.RunWindow)
        runWindowThread.start()
        
        while(True):
            
            # Capture the video frame
            # by frame
            ret, frame = vid.read()

            # I can start a thread here that processes the frames
        
            # Display the resulting frame
            cv2.imshow('title', frame[:,:,0])
            
            # the 'q' button is set as the
            # quitting button you may use any
            # desired button of your choice
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # break 
        
        # After the loop release the cap object
        vid.release()
        # Destroy all the windows
        cv2.destroyAllWindows()

    def Test(self):
        index = 4
        print(self._labelDictionary[int(self._testData._dataSet[self._testData._targetColumns].loc[index])])
        preds = self._model.predict(self._xTest)
        print(preds)
    
    def RunWindow(self):
        window = tk.Tk()
        greeting = tk.Label(text="Hello, Tkinter")
        greeting.pack()
        window.mainloop()

class SampleApp(tk.Tk):

    def __init__(self):
        tk.Tk.__init__(self)
        self.label = tk.Label(self, text='Enter text')
        self.label.pack(side = 'top', pady = 5)
        self.button = tk.Button(self, text='update', command=self.on_button)
        self.button.pack()

    def on_button(self):
        self.label['text'] +='\nNew New Text'


# class Project():
#     """
#     Final Project
#     =============
#     Author: Brando 

#     Oultine 
#     -------------
#     The final image recognition project is following the structure of Chapter 2, that is the main steps to consider are:

#     1. Look at the big picture.
#     2. Get the data.
#     3. Discover and visualize the data to gain insights.
#     4. Prepare the data for Machine Learning algorithms.
#     5. Select a model and train it.
#     6. Fine-tune your model.

#     Depending on the chosen project, It is not expected to complete the task that you set out to do but it is expected that at least one of the above steps challenges you, i.e. is different from what we have done in class. For example, getting the data could be challenging because it is not straightforward to scrape the images from the web, building a model could be changing the loss function of a neural network that is specific to the project.

#     The finals week of class will be presenting the results of your project in a similar form to an article. In particular,

#     - Intoduction. What is the problem you are trying to solve?

#     - Dataset. present the details of dataset, displaying example images and how you obtained the dataset

#     - Fine-Tuning. The different model parameters that were attempted

#     - Results including accuracy tables and display the final outputs, i.e. how the trained network performed on some sample images

#     - Outlook, discuss points of improvement or future steps to take the project further.

#     I will be asking questions during the presentation to confirm your understanding in each step in the process.

#     Details
#     -------------
#     Breaking down the outline:

#     1. Look at the big picture.

#     It is more motivating to choose a project that you personally find practical or that fills an obvious need.

#     2. Get the data.

#     Google Images has already done some hard work in labelling items but it is a good starting point if you need to get classified images. Note that scraping can be a challenge in itself, there is a bit of a hack using the package pyautogui, where you can give control of the keyword and mouse/cursor to the computer and perform the manual actions that you could take to save images from image search.

#     There will be a challenge of creating a big enough dataset that is relatively unbiased. As you have observed, more examples lead to greater accuracy but gathering data can also be a big time suck so once you feel there is enough to continue with the following steps, then come back and gather even more data when you have prepared the code for the rest of the project.

#     3. Discover and visualize the data to gain insights

#     This stage of Exploratory Data Analysis (EDA) could be just a matter of viewing the images and the targets. If the images were scraped from Google Images, there may be some false positives and it would be good to crop them out. In the FacialKeypoints, the missing data there shared how unclean the data was but if you start to look at specific images, e.g. the shortest distance between eyes, you will find that there are many faces that didn't fill the whole 96x96 grid and may have skewed the learning/accuracy. Another reason to perform this step and be comfortable that the quality of the data going into the model is good.

#     4& 5. Select a model, Prepare the data for Machine Learning algorithms. Train the model

#     Steps 4 and 5 in the outline are not separable for me, i.e. you cannot prepare the data if you don't know the model that you are trying to fit or the loss you are trying to optimize.

#     For example, the Yolo network has a very particular output/target and you would need to prepare the data when you have understood the neural network output.

#     6. Fine-tune your model.

#     Applying hyperparameter optimization to improve your accuracy. This shouldn't be too challenging because you should be reimplementing the code that was prepared for Week 7 Assignment.

#     Example projects:

#     - Street2Shop, matching an image of clothing off the street with a similar item in an online store

#     - Grape Detection, identifying the location and number of grapes from an image

#     - Make/Model Classification, classifying the make and model of a car from whatever angle

#     - OCR (Optical Character Recognition), being able to convert the image of text into the string

#     - Sky segmentation, being able to identify the sky in an image.

#     - Helmet Detection, identifying whether motorcyclists or cyclists are wearing helmets.
#     Overview 
#     -------------
#     My project is inspired by https://www.youtube.com/watch?v=pDXdlXlaCco 
#     Dataset: https://www.kaggle.com/datamunge/sign-language-mnist

#     """

#     _basePath = sys.path[0]

#     def __init__(self) -> None:
#         self._signLangHandler = xASLHandler(1)
    
#     def Do(self):
#         self._signLangHandler.Run()

def main():
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
    # w = SampleApp()
    # w.resizable(width=True, height=True)
    # w.geometry('{}x{}'.format(100, 90))
    # w.label.config(text=w.label['text']+'\nnew text')
    # w.mainloop()

    signLangHandler = xASLHandler(1)
    signLangHandler.Run()

if __name__ == "__main__":
    # print(Project.__doc__)

    # project = Project()

    # project.Do()
    main()
    
