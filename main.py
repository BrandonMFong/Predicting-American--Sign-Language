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

class TextWindow(tk.Tk):
    """
    https://stackoverflow.com/questions/45397806/update-text-on-a-tkinter-window
    """
    def __init__(self):
        tk.Tk.__init__(self)
        self.label = tk.Label(self, text='Enter text')
        self.label.pack(side = 'top', pady = 5)
        self.button = tk.Button(self, text='update', command=self.on_button)
        self.button.pack()
        self._number = 0

    def on_button(self):
        self.label['text'] ="{}".format(self._number)
        self._number += 1

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

        runWindowThread.join()

    def Test(self):
        index = 4
        print(self._labelDictionary[int(self._testData._dataSet[self._testData._targetColumns].loc[index])])
        preds = self._model.predict(self._xTest)
        print(preds)
    
    def RunWindow(self):
        w = TextWindow()
        w.resizable(width=True, height=True)
        w.geometry('{}x{}'.format(100, 90))
        w.mainloop()

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

    """
    signLangHandler = xASLHandler(1)
    signLangHandler.Run()

if __name__ == "__main__":
    main()
    
