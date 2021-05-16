"""
main.py
================
Contains xASLHandler class

Todo
-------------------
- Properly exit out of application 
"""
# Brando 
# 3/30/2021
# Final Project

from    tensorflow                              import keras
from    atpbar                                  import atpbar
from    Library                                 import FunctionLibrary  as fLib
from    Library                                 import FileSystem       as fs 
from    Library                                 import Logger, InitError, Base, YES, NO, GetPercentage
from    tensorflow.keras.preprocessing.image    import ImageDataGenerator
from    sklearn.model_selection                 import train_test_split
from    tensorflow.keras.utils                  import to_categorical
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

class TextWindow(tk.Tk):
    """
    https://stackoverflow.com/questions/45397806/update-text-on-a-tkinter-window
    """
    def __init__(self, stopRecordingCallback):
        tk.Tk.__init__(self)

        # Window killed flag

        # Create prediction label 
        self._predictionLabel = tk.Label(self, text='Prediction')
        self._predictionLabel.pack(side = 'top', pady = 5)

        # Create Confidence label 
        self._confidenceLabel = tk.Label(self, text='Confidence')
        self._confidenceLabel.pack(side = 'top', pady = 5)

        # Stop button 
        self.button = tk.Button(self, text='stop', command=self.on_button)
        self.button.pack()

        # Save callback function
        self._stopRecordingCallback = stopRecordingCallback

    def on_button(self):
        self.destroy()
        self._stopRecordingCallback()

class xASLHandler():
    """
    xASLHandler
    =============
    """

    # Data
    _rawTestFile    = "data/test/sign.csv"
    _rawTrainFile   = "data/train/sign.csv"

    # Cache files
    _testCache  = "testImages.cache"
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

    _testArrayName  = "test"
    _trainArrayName = "train"

    def __init__(self,epochs=10) -> None:
        okayToContinue      = True 
        fullTestFilename    = None
        fullTrainFilename   = None
        temp                = None
        doTrain             = YES

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
        self._log               = Logger(scriptName=__file__)
        self._textWindow        = None
        self._keepRecording     = True

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
            else:
                self._log.Error("Incompatable size for input images.  Total pixels for dataset:", len(self._trainData._trainColumns))
                okayToContinue = False

        # Initialize the model 
        if okayToContinue:
            self.GetIOs2()
            okayToContinue = self.CreateModel2()

        # Train the model 
        if okayToContinue and doTrain:
            self.Train2()

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

    def GetIOs2(self):
        """
        https://www.kaggle.com/hkubra/mnist-cnn-with-keras-99-accuracy
        """
        self._xTrain    = self._imageTrainArray.astype(float)
        self._yTrain    = self._trainData._dataSet[self._trainData._targetColumns]
        self._xTest     = self._imageTestArray.astype(float)
        self._yTest     = self._testData._dataSet[self._testData._targetColumns].astype(float)

        self._yTrain    = to_categorical(self._yTrain, num_classes=25)
        self._yTest     = to_categorical(self._yTest, num_classes=25)

        self._xTrain    = self._xTrain/255.0
        self._xTest     = self._xTest/255.0

    def CreateModel2(self):
        success = True 
        try:
            # Establish Sequential model 
            self._model = keras.Sequential()

            # Conv2D
            self._model.add(
                keras.layers.Conv2D(
                    filters     = 8, 
                    kernel_size = (5,5),
                    padding     = 'Same', 
                    activation  = 'relu', 
                    input_shape = self._reshapeValue
                    # input_shape = (28,28,1)
            ))

            # MaxPool2D
            self._model.add(keras.layers.MaxPool2D(pool_size=(2,2)))

            self._model.add(keras.layers.Dropout(0.25))

            # Conv2D
            self._model.add(
                keras.layers.Conv2D(
                    filters = 16, 
                    kernel_size = (3,3),
                    padding = 'Same', 
                    activation ='relu'
            ))

            # MaxPool2D
            self._model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))

            self._model.add(keras.layers.Dropout(0.25))

            self._model.add(keras.layers.Flatten())

            self._model.add(keras.layers.Dense(512, activation = "relu"))

            self._model.add(keras.layers.Dropout(0.5))

            # 25 outputs
            self._model.add(keras.layers.Dense(25, activation = "softmax"))

            # Adam optimizer 
            optimizer = keras.optimizers.Adam(
                lr      = 0.001, 
                beta_1  = 0.9,
                beta_2  = 0.999
            )

            self._model.compile(
                optimizer   = optimizer , 
                loss        = "categorical_crossentropy", 
                metrics     = ["accuracy"]
            )

            # Print Summary
            self._model.summary()

        except Exception as e:
            self._log.Except(e)
            success = False 
        return success

    def Train2(self):
        """
        https://www.kaggle.com/hkubra/mnist-cnn-with-keras-99-accuracy
        """
        batch_size = 128
        datagen = ImageDataGenerator(
            featurewise_center              = False, 
            samplewise_center               = False, 
            featurewise_std_normalization   = False, 
            samplewise_std_normalization    = False, 
            zca_whitening                   = False, 
            rotation_range                  = 10, 
            zoom_range                      = 0.1, 
            width_shift_range               = 0.1, 
            height_shift_range              = 0.1,
            horizontal_flip                 = False, 
            vertical_flip                   = False
        ) 

        datagen.fit(self._xTrain)
        _ = self._model.fit_generator(
            datagen.flow(
                self._xTrain, 
                self._yTrain, 
                batch_size = batch_size
            ),
            epochs          = self._epochs, 
            validation_data = (self._xTest, self._yTest),
            steps_per_epoch = self._xTrain.shape[0] // batch_size
        )

    def Record(self):
        """
        Run 
        ================
        """
        self._keepRecording = True 

        # define a video capture object
        vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        # runWindowThread         = threading.Thread(target=self.RunWindow)
        # runWindowThread.daemon  = True 
        # runWindowThread.start()
        
        try:
            while self._keepRecording:
                
                # Capture the video frame
                # by frame
                _, frame = vid.read()

                # I can start a thread here that processes the frames
            
                # Display the resulting frame
                pred, conf = self.GetPrediction(frame)
                self.UpdateText(pred,conf)
                cv2.imshow('title', frame)
                
                # the 'q' button is set as the
                # quitting button you may use any
                # desired button of your choice
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except KeyboardInterrupt:
            pass 
        
        # After the loop release the cap object
        vid.release()
        # Destroy all the windows
        cv2.destroyAllWindows()
        # runWindowThread.join()

    def GetPrediction(self,frame: np) -> str:
        result = str() 
        conf = float()

        # Resize 
        res = cv2.resize(frame,(28,28),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
        res = np.expand_dims(res[:,:,0], axis=-1)
        res = np.expand_dims(res, axis=0)
        res = res.astype(float)

        # Normalize 
        inputData   = ImageDataGenerator(rescale=1/255)
        input       = inputData.flow(res)

        # Predict
        array   = self._model.predict(input)
        index   = np.argmax(array)
        conf    = array[0][index]
        result  = self._labelDictionary[index]

        return result, conf

    def Run(self):
        # Start the recording thread 
        recordThread         = threading.Thread(target=self.Record)
        recordThread.daemon  = True 
        recordThread.start()

        # Initialize and start the output window 
        self._textWindow = TextWindow(stopRecordingCallback=self.StopRecording)
        self._textWindow.resizable(width=True, height=True)
        self._textWindow.geometry('{}x{}'.format(200, 200))
        self._textWindow.mainloop()

    def UpdateText(self, value: str, confidence: float):
        if self._textWindow is not None: 
            self._textWindow._predictionLabel['text'] = "Prediction: {}".format(value)
            self._textWindow._confidenceLabel['text'] = "Confidence: {}%".format(GetPercentage(confidence))

    def StopRecording(self):
        self._keepRecording = False 

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
    signLangHandler = xASLHandler(epochs=1)
    signLangHandler.Run()

if __name__ == "__main__":
    main()
    
